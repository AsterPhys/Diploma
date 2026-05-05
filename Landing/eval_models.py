import os
import cv2
import json
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
from tqdm import tqdm
from thop import profile

import config
import config_train
import segmentation_models_pytorch as smp
from ultralytics import YOLO

import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_smp_model_func(model_info, device):
	with open(model_info["config"], "r", encoding="utf-8") as f:
		cfg = json.load(f)
	
	model_cfg = cfg["model"]
	extra_kwargs = cfg.get("extra_kwargs", {})
	if isinstance(extra_kwargs, str):
		extra_kwargs = json.loads(extra_kwargs)
		
	model_class = getattr(smp, model_cfg["architecture"])
	model = model_class(
		encoder_name=model_cfg["backbone"],
		encoder_weights=None,
		in_channels=3,
		classes=model_cfg["num_classes"],
		**extra_kwargs
	)
	
	checkpoint = torch.load(model_info["weights"], map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()
	return model, cfg

def load_yolo_model_func(model_info, device):
	model = YOLO(model_info["weights"])
	return model, None

# ======== ДАТАСЕТ ДЛЯ ОЦЕНКИ ========
class EvalDataset(torch.utils.data.Dataset):
	def __init__(self, rgb_dir, mask_dir, img_names):
		self.rgb_dir = rgb_dir
		self.mask_dir = mask_dir
		self.images = img_names

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_name = self.images[idx]
		img_path = os.path.join(self.rgb_dir, img_name)
		image_bgr = cv2.imread(img_path)
		if image_bgr is None:
			raise ValueError(f"Не удалось прочитать изображение: {img_path}")
		
		mask_path = os.path.join(self.mask_dir, img_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		
		new_mask = np.zeros_like(mask, dtype=np.int32)
		new_mask[mask == 5] = 0                    # Safe_Ground
		new_mask[(mask == 1) | (mask == 3)] = 1    # Static_Obstacle
		new_mask[mask == 4] = 2                    # Vegetation
		new_mask[mask == 2] = 3                    # Dynamic_Obstacle
		
		return image_bgr, new_mask

# ======== ПОИСК ВСЕХ ОБУЧЕННЫХ МОДЕЛЕЙ ========
def find_trained_models(base_dir):
	"""Рекурсивно ищет модели SMP и YOLO в указанной директории."""
	models = []
	for root, dirs, files in os.walk(base_dir):
		# Проверка на модель SMP
		if "best_model.pth" in files and "config.json" in files:
			run_name = os.path.relpath(root, base_dir).replace("\\", "/")
			models.append({
				"type": "smp",
				"name": run_name,
				"path": root,
				"weights": os.path.join(root, "best_model.pth"),
				"config": os.path.join(root, "config.json")
			})
		# Проверка на модель YOLO (ищет best.pt)
		elif "best.pt" in files:
			run_name = os.path.relpath(root, base_dir).replace("\\", "/")
			if os.path.basename(root) == "weights":
				run_name = os.path.relpath(os.path.dirname(root), base_dir).replace("\\", "/")
			models.append({
				"type": "yolo",
				"name": run_name,
				"path": root,
				"weights": os.path.join(root, "best.pt"),
				"config": None
			})
	return models

# ======== ОЦЕНКА МАКСОВ И ПАРАМЕТРОВ ========
def get_model_stats(model, model_info, device):
	dummy_input = torch.randn(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
	macs_g = 0.0
	params_m = 0.0
	
	try:
		if model_info["type"] == "smp":
			macs, params = profile(model, inputs=(dummy_input,), verbose=False)
			macs_g = round(macs / 1e9, 4)
			params_m = round(params / 1e6, 4)
		else: # YOLO
			info = model.info(verbose=False)
			params_m = round(info[1] / 1e6, 4)
			# info[3] содержит GFLOPs. Так как 1 MAC +- 2 FLOPs, делим на 2
			macs_g = round(info[3] / 2.0, 4) if isinstance(info[3], (int, float)) else -1.0
	except Exception as e:
		print(f"[!] Ошибка замера сложности для {model_info['name']}: {e}")
		macs_g, params_m = -1.0, -1.0
		
	return macs_g, params_m

# ======== ИНФЕРЕНС И РАСЧЕТ МЕТРИК ========
def evaluate_model(model, model_info, dataset, device):
	keys = list(range(config_train.NUM_CLASSES)) + ["any_obstacle"]
	total_metrics = {k: {"intersection": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0} for k in keys}
	
	eval_transform = A.Compose([
		A.Normalize(),
		ToTensorV2()
	])
	
	total_time_ms = 0.0
	total_frames = 0
	is_smp = model_info["type"] == "smp"
	device_str = "cuda:0" if device.type == "cuda" else "cpu"
	
	# Прогревочный прогон для точности замера времени
	print(f"[*] Прогрев модели {model_info['name']}...")
	for _ in range(5):
		dummy = torch.randn(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
		if is_smp:
			_ = model(dummy)
		else:
			# Для YOLO прогрев на пустом кадре
			dummy = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=np.uint8)
			_ = model(dummy, verbose=False, device=device_str, imgsz=[config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
			
	# Основной цикл оценки
	for idx in tqdm(range(len(dataset)), desc=f"Оценка {model_info['name']}"):
		image_bgr, true_mask = dataset[idx]
		h, w = image_bgr.shape[:2]
		
		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		
		if is_smp:
			image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
			augmented = eval_transform(image=image_rgb)
			smp_tensor = augmented['image'].unsqueeze(0).to(device)
			
			start_event.record()
			with torch.no_grad():
				outputs = model(smp_tensor)
				pred_mask_tensor = torch.argmax(outputs, dim=1).squeeze(0)
			end_event.record()
			
			torch.cuda.synchronize()
			pred_mask = pred_mask_tensor.cpu().numpy()
			
		else: # YOLO
			device_str = "cuda:0" if device.type == "cuda" else "cpu"
			
			start_event.record()
			results = model(image_bgr, verbose=False, device=device_str, imgsz=[480, 640])
			end_event.record()
			
			torch.cuda.synchronize()
			
			# Постпроцессинг: восстанавливаем 2D семантическую маску из детекций
			pred_mask = np.zeros((h, w), dtype=np.uint8) # Фон 0 - Safe Ground
			result = results[0]
			if result.masks is not None:
				masks = result.masks.data.cpu().numpy()
				classes = result.boxes.cls.cpu().numpy().astype(int)
				confs = result.boxes.conf.cpu().numpy()
				
				# Перекрываем маски по увеличению конфиденса
				sorted_idx = np.argsort(confs)
				for s_idx in sorted_idx:
					cls = classes[s_idx]
					mask = masks[s_idx]
					mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
					pred_mask[mask_resized > 0.5] = cls
					
		# Накапливаем задержку (Latency)
		total_time_ms += start_event.elapsed_time(end_event)
		total_frames += 1
		
		pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).to(device)
		true_tensor = torch.from_numpy(true_mask).unsqueeze(0).to(device)
		
		for cls in range(config_train.NUM_CLASSES):
			pred_inds = (pred_tensor == cls)
			target_inds = (true_tensor == cls)
			
			intersection = (pred_inds & target_inds).sum().item()
			union = (pred_inds | target_inds).sum().item()
			fp = (pred_inds & ~target_inds).sum().item()
			fn = (~pred_inds & target_inds).sum().item()
			
			total_metrics[cls]["intersection"] += intersection
			total_metrics[cls]["union"] += union
			total_metrics[cls]["tp"] += intersection
			total_metrics[cls]["fp"] += fp
			total_metrics[cls]["fn"] += fn
			
		# Любое препятствие (классы > 0)
		pred_danger = (pred_tensor > 0)
		target_danger = (true_tensor > 0)
		
		danger_intersection = (pred_danger & target_danger).sum().item()
		danger_union = (pred_danger | target_danger).sum().item()
		danger_fp = (pred_danger & ~target_danger).sum().item()
		danger_fn = (~pred_danger & target_danger).sum().item()
		
		total_metrics["any_obstacle"]["intersection"] += danger_intersection
		total_metrics["any_obstacle"]["union"] += danger_union
		total_metrics["any_obstacle"]["tp"] += danger_intersection
		total_metrics["any_obstacle"]["fp"] += danger_fp
		total_metrics["any_obstacle"]["fn"] += danger_fn
		
	# Расчет итоговых показателей
	mean_ious = {}
	for key in keys:
		union = total_metrics[key]["union"]
		intersection = total_metrics[key]["intersection"]
		mean_ious[key] = intersection / union if union > 0 else 0.0
		
	valid_class_ious = [mean_ious[i] for i in range(config_train.NUM_CLASSES) if total_metrics[i]["union"] > 0]
	mIoU = np.mean(valid_class_ious) if valid_class_ious else 0.0
	
	tp_sg, fp_sg = total_metrics[0]["tp"], total_metrics[0]["fp"]
	safe_ground_precision = tp_sg / (tp_sg + fp_sg) if (tp_sg + fp_sg) > 0 else 0.0
	
	tp_obs, fn_obs = total_metrics["any_obstacle"]["tp"], total_metrics["any_obstacle"]["fn"]
	any_obstacle_recall = tp_obs / (tp_obs + fn_obs) if (tp_obs + fn_obs) > 0 else 0.0
	
	avg_latency_ms = total_time_ms / total_frames if total_frames > 0 else 0.0
	fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
	
	return {
		"mIoU": round(mIoU, 4),
		"safe_ground_iou": round(mean_ious[0], 4),
		"safe_ground_precision": round(safe_ground_precision, 4),
		"any_obstacle_recall": round(any_obstacle_recall, 4),
		"latency_ms": round(avg_latency_ms, 2),
		"fps": round(fps, 2),
		"iou_static_obstacle": round(mean_ious[1], 4),
		"iou_vegetation": round(mean_ious[2], 4),
		"iou_dynamic_obstacle": round(mean_ious[3], 4),
		"iou_any_obstacle": round(mean_ious["any_obstacle"], 4)
	}

# ======== ВИЗУАЛИЗАЦИЯ И СТРОИТЕЛЬСТВО ГРАФИКОВ ========
def plot_results(results, output_dir):
	os.makedirs(output_dir, exist_ok=True)
	
	names = [r["name"] for r in results]
	mIou = [r["mIoU"] for r in results]
	precision = [r["safe_ground_precision"] for r in results]
	recall = [r["any_obstacle_recall"] for r in results]
	fps = [r["fps"] for r in results]
	types = [r["type"] for r in results]
	
	# --- График 1: Столбчатая диаграмма метрик точности и безопасности ---
	x = np.arange(len(names))
	width = 0.25
	
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.bar(x - width, mIou, width, label='mIoU (Среднее по классам)', color='#00cec9')
	ax.bar(x, precision, width, label='Safe Ground Precision (Безопасность)', color='#0984e3')
	ax.bar(x + width, recall, width, label='Any Obstacle Recall (Обнаружение)', color='#d63031')
	
	ax.set_ylabel('Оценка (0.0 - 1.0)')
	ax.set_title('Сравнение моделей: Точность Сегментации & Показатели Безопасности')
	ax.set_xticks(x)
	ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
	ax.set_ylim(0, 1.05)
	ax.legend(loc='lower left')
	ax.grid(axis='y', linestyle='--', alpha=0.5)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "metrics_comparison_bar.png"), dpi=150)
	plt.close()
	
	# --- График 2: Диаграмма Парето (Скорость vs Безопасность) ---
	plt.figure(figsize=(10, 6))
	
	for i in range(len(results)):
		color = '#ff7675' if types[i] == 'yolo' else '#74b9ff'
		marker = 's' if types[i] == 'yolo' else 'o'
		plt.scatter(fps[i], precision[i], s=130, color=color, marker=marker, edgecolors='black', zorder=3)
		plt.text(fps[i] + 0.8, precision[i] + 0.002, names[i], fontsize=8, zorder=4)
		
	plt.xlabel('Скорость работы (FPS)')
	plt.ylabel('Точность поиска безопасной зоны (Safe Ground Precision)')
	plt.title('Парето-оптимизация: Баланс Скорости (FPS) и Безопасности посадки')
	plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
	
	legend_elements = [
		Line2D([0], [0], marker='o', color='w', label='Модели PyTorch (SMP)', markerfacecolor='#74b9ff', markersize=10, markeredgecolor='black'),
		Line2D([0], [0], marker='s', color='w', label='Модели YOLOv11', markerfacecolor='#ff7675', markersize=10, markeredgecolor='black')
	]
	plt.legend(handles=legend_elements, loc='lower left')
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "speed_vs_safety_pareto.png"), dpi=150)
	plt.close()
	print(f"[*] Графики успешно сохранены в папку: {output_dir}")

# ======== 6. ВЫВОД КРАСИВОЙ ТАБЛИЦЫ ========
def print_results_table(results):
	cols = ["Model Name", "Type", "mIoU", "SG IoU", "SG Prec", "Obs Rec", "Latency (ms)", "FPS", "Params (M)", "MACs (G)"]
	row_fmt = "{:<32} | {:<5} | {:<6} | {:<6} | {:<7} | {:<7} | {:<12} | {:<6} | {:<10} | {:<8}"
	
	print("\n" + "="*117)
	print(" " * 44 + "ИТОГОВЫЙ СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
	print("="*117)
	print(row_fmt.format(*cols))
	print("-"*117)
	
	for r in results:
		print(row_fmt.format(
			r["name"][:32],
			r["type"],
			f"{r['mIoU']:.4f}",
			f"{r['safe_ground_iou']:.4f}",
			f"{r['safe_ground_precision']:.4f}",
			f"{r['any_obstacle_recall']:.4f}",
			f"{r['latency_ms']:.2f}",
			f"{r['fps']:.1f}",
			f"{r['params_m']:.2f}",
			f"{r['macs_g']:.2f}" if r["macs_g"] >= 0 else "N/A"
		))
	print("="*117 + "\n")

# ======== ОСНОВНОЙ КОД ЗАПУСКА ========
def main():
	parser = argparse.ArgumentParser(description="Скрипт детальной оценки обученных сетей сегментации и YOLO")
	parser.add_argument('--session', type=str, default=None, help="Имя папки Grid Search сессии (если нужно оценить только ее)")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[*] Устройство для проведения оценки: {device}")

	test_dir = config.TEST_DATA_DIR

	# Выделение тестового множества кадров
	print(f"[*] Загрузка выделенного тест-сета из {test_dir}...")
	rgb_dir = os.path.join(test_dir, "rgb")
	mask_dir = os.path.join(test_dir, "mask")
	test_names = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
		
	dataset = EvalDataset(rgb_dir, mask_dir, test_names)
	print(f"[УСПЕХ] Датасет инициализирован. Всего кадров для оценки: {len(dataset)}\n")

	# Поиск моделей для тестирования
	scan_dir = config.MODELS_DIR
	if args.session:
		scan_dir = os.path.join(config.MODELS_DIR, args.session)
		
	print(f"[*] Сканирование директории {scan_dir} на наличие обученных моделей...")
	found_models = find_trained_models(scan_dir)
	
	if not found_models:
		print("[!] Не найдено ни одной обученной модели. Убедитесь, что обучение завершено и файлы .pth/.pt сохранены.")
		return
		
	print(f"[УСПЕХ] Обнаружено моделей для анализа: {len(found_models)}")
	
	results = []
	
	# Цикл оценки моделей
	for m_info in found_models:
		print(f"\n{'-'*60}")
		print(f"[*] Начало работы с моделью: {m_info['name']} ({m_info['type'].upper()})")
		print(f"{'-'*60}")
		
		try:
			if m_info["type"] == "smp":
				model, _ = load_smp_model_func(m_info, device)
			else:
				model, _ = load_yolo_model_func(m_info, device)
		except Exception as e:
			print(f"[ОШИБКА] Не удалось загрузить веса для {m_info['name']}: {e}. Пропускаем...")
			continue
			
		# Считаем MACs и параметры
		macs_g, params_m = get_model_stats(model, m_info, device)
		
		# Оцениваем качество и FPS
		metrics = evaluate_model(model, m_info, dataset, device)
		
		# Объединяем результаты
		metrics["name"] = m_info["name"]
		metrics["type"] = m_info["type"]
		metrics["macs_g"] = macs_g
		metrics["params_m"] = params_m
		
		results.append(metrics)
		
		del model
		torch.cuda.empty_cache()

	if not results:
		print("[!] Оценка не выдала результатов.")
		return

	# Сортируем результаты по mIoU от лучшей к худшей
	results.sort(key=lambda x: x["mIoU"], reverse=True)

	out_dir = os.path.join(scan_dir, "evaluation_results")
	os.makedirs(out_dir, exist_ok=True)
	
	print_results_table(results)
	
	csv_path = os.path.join(out_dir, "summary_results.csv")
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=results[0].keys())
		writer.writeheader()
		writer.writerows(results)
		
	json_path = os.path.join(out_dir, "summary_results.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=4, ensure_ascii=False)
		
	print(f"[*] Таблицы результатов сохранены в:\n  -> {csv_path}\n  -> {json_path}")
	
	plot_results(results, out_dir)

if __name__ == "__main__":
	main()