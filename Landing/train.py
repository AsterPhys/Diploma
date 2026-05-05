import os
import cv2

# Настройки и фиксы для обучения на AMD
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
# os.environ['HSA_XNACK'] = '0'

os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'

os.environ['MIOPEN_USER_DB_PATH'] = 'D:\\Cache\\Temp'
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = 'D:\\Cache\\Temp'

import json
import torch
import numpy as np
from tqdm import tqdm
import glob
import re
from thop import profile

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import segmentation_models_pytorch as smp

import config
import config_train

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

DEBUG = False

# ----------------------------------

def paint_segmap(mask):
	"""Получает из маски сегментации с классами маску с цветами."""
	if torch.is_tensor(mask):
		mask = mask.cpu().numpy()
	rgb = config_train.COLOR_MAP[mask]
	# Транспонируем для TensorBoard (HWC -> CHW).
	return np.transpose(rgb, (2, 0, 1))

def get_intersection_and_union(pred_masks, true_masks, num_classes):
	"""
	Возвращает значения Intersection, Union, TP, FP и FN для каждого класса.
	"""
	preds = torch.argmax(pred_masks, dim=1)
	
	metrics = {}
	for cls in range(num_classes):
		pred_inds = (preds == cls)
		target_inds = (true_masks == cls)

		intersection = (pred_inds & target_inds).sum().item() # Это же и TP
		union = (pred_inds | target_inds).sum().item()
		fp = (pred_inds & ~target_inds).sum().item()
		fn = (~pred_inds & target_inds).sum().item()
		
		metrics[cls] = {
			"intersection": intersection,
			"union": union,
			"tp": intersection,
			"fp": fp,
			"fn": fn
		}

	# Бинарный IoU (Safe_Ground vs Все остальные классы - препятствия)
	pred_danger = (preds > 0)
	target_danger = (true_masks > 0)

	danger_intersection = (pred_danger & target_danger).sum().item()
	danger_union = (pred_danger | target_danger).sum().item()
	danger_fp = (pred_danger & ~target_danger).sum().item()
	danger_fn = (~pred_danger & target_danger).sum().item()

	metrics["any_obstacle"] = {
		"intersection": danger_intersection, 
		"union": danger_union,
		"tp": danger_intersection,
		"fp": danger_fp,
		"fn": danger_fn
	}

	return metrics

class DroneLandingDataset(Dataset):
	def __init__(self, rgb_dir, mask_dir, img_names, transform=None):
		self.rgb_dir = rgb_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.images = img_names

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if DEBUG:
			print(f"DEBUG: Загружаю картинку {idx}")
		img_name = self.images[idx]
		
		# Загружаем RGB
		img_path = os.path.join(self.rgb_dir, img_name)
		image = cv2.imread(img_path)
		if image is None:
			raise ValueError(f"Не удалось прочитать изображение: {img_path}")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		# Загружаем маску
		mask_path = os.path.join(self.mask_dir, img_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Читаем как одноканальную

		# Меняем нумерацию классов:
		# делаем так, чтобы она начиналась с 0 + Hazard --> Static_Obstacle
		# (мало объектов этого класса на карте)
		new_mask = np.zeros_like(mask, dtype=np.int32)
		
		new_mask[mask == 5] = 0					# Safe_Ground
		new_mask[(mask == 1) | (mask == 3)] = 1 # Static_Obstacle
		new_mask[mask == 4] = 2					# Vegetation
		new_mask[mask == 2] = 3					# Dynamic_Obstacle

		# Применяем аугментации
		if self.transform:
			augmented = self.transform(image=image, mask=new_mask)
			image = augmented['image']
			new_mask = augmented['mask']
		
		return image, new_mask.long()

def train_model():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# torch.backends.cudnn.benchmark = True
	print(f"[TRAIN] Используемое устройство для обучения: {device}")

	run_name = os.environ.get("RUN_NAME", "Default_Run")
	run_dir = os.path.join(config.MODELS_DIR, run_name)
	os.makedirs(run_dir, exist_ok=True)
	print(f"[TRAIN] Эксперимент: {run_name}")

	config_dict = config_train.get_train_config_dict()
	with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config_dict, f, indent=4)

	tensorboard_dir = os.path.join(run_dir, "logs")
	writer = SummaryWriter(log_dir=tensorboard_dir)

	# ======== Подготовка данных ========
	all_images = sorted(os.listdir(config.RGB_DIR))
	np.random.shuffle(all_images)

	val_size = int(len(all_images) * 0.2)
	train_names = all_images[val_size:]
	val_names = all_images[:val_size]

	train_dataset = DroneLandingDataset(
		config.RGB_DIR, config.MASK_DIR, 
		img_names=train_names, 
		transform=config_train.TRAIN_TRANSFORMS
	)
	
	val_dataset = DroneLandingDataset(
		config.RGB_DIR, config.MASK_DIR, 
		img_names=val_names, 
		transform=config_train.VAL_TRANSFORMS
	)

	# Сохраняем примеры изображений
	first_img, _ = train_dataset[0]
	target_size = first_img.shape

	example_images = []
	example_masks = []
	i, k = 0, 0
	while k < config_train.N_SAVE_EXAMPLES and i < len(train_dataset):
		img, mask = train_dataset[i]

		if img.shape == target_size:
			example_images.append(img)

			colored_mask = paint_segmap(mask)
			mask_tensor = torch.from_numpy(colored_mask).float()
			mask_tensor /= 255.0
			example_masks.append(mask_tensor)

			k += 1
		i += 1

	img_grid = torchvision.utils.make_grid(example_images, nrow=4)
	mask_grid = torchvision.utils.make_grid(example_masks, nrow=4)

	writer.add_image("Dataset/Images", img_grid)
	writer.add_image("Dataset/Masks_Colored", mask_grid)
	
	writer.flush()

	train_loader = DataLoader(
		train_dataset, 
		batch_size=config_train.BATCH_SIZE, 
		shuffle=True, 
		num_workers=config_train.NUM_WORKERS, 
		drop_last=True, 
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset, 
		batch_size=config_train.BATCH_SIZE, 
		shuffle=False, 
		num_workers=config_train.NUM_WORKERS, 
		pin_memory=True
	)

	# ======== Инициализация модели ========
	model_class = getattr(smp, config_train.SEG_MODEL_NAME)
	model = model_class(
		encoder_name=config_train.SEG_BACKBONE,
		encoder_weights=config_train.SEG_WEIGHTS,
		in_channels=3,
		classes=config_train.NUM_CLASSES,
		**config_train.EXTRA_KWARGS
	).to(device)

	if config_train.OPTIMIZER == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=config_train.LEARNING_RATE)
	elif config_train.OPTIMIZER == "AdamW":
		optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.LEARNING_RATE, weight_decay=1e-4)
	elif config_train.OPTIMIZER == "SGD":
		optimizer = torch.optim.SGD(model.parameters(), lr=config_train.LEARNING_RATE, momentum=0.9)

	if config_train.CRITERION == "CrossEntropyLoss":
		criterion = nn.CrossEntropyLoss()

	# ======== Оценка алгоритмической сложности модели и подготовка метрик ========
	metrics_file = os.path.join(run_dir, "metrics.json")
	metrics_data = {"model_stats": {"macs_g": 0.0, "params_m": 0.0}, "history":[]}

	dummy_input = torch.randn(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
	macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

	metrics_data["model_stats"]["macs_g"] = round(macs / 1e9, 4)
	metrics_data["model_stats"]["params_m"] = round(params / 1e6, 4)
		
	print(f"\n[MODEL STATS] Архитектура: {config_train.SEG_MODEL_NAME} | Backbone: {config_train.SEG_BACKBONE}")
	print(f"[MODEL STATS] Вычислительная сложность (MACs): {metrics_data['model_stats']['macs_g']} G")
	print(f"[MODEL STATS] Количество параметров: {metrics_data['model_stats']['params_m']} M\n")

	# ======== Восстановление из чекпоинта ========
	start_epoch = 0
	best_val_iou = 0.0

	checkpoint_files = glob.glob(os.path.join(run_dir, "model_epoch_*.pth"))
	if checkpoint_files:
		print(f"\n[RESUME] Найдено прерванное обучение")
		def extract_epoch(filepath):
			match = re.search(r'model_epoch_(\d+)\.pth', os.path.basename(filepath))
			return int(match.group(1)) if match else -1

		latest_checkpoint = max(checkpoint_files, key=extract_epoch)
		checkpoint = torch.load(latest_checkpoint, map_location=device)

		print(f"[RESUME] Загружаю веса из: {latest_checkpoint}")

		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		best_val_iou = checkpoint.get('best_val_iou', 0.0)

		# --- Подгружаем историю метрик ---
		if os.path.exists(metrics_file):
			with open(metrics_file, "r", encoding="utf-8") as f:
				try:
					loaded_metrics = json.load(f)
					# Оставляем историю только до той эпохи, с которой восстанавливаемся
					loaded_metrics["history"] = [h for h in loaded_metrics.get("history", []) if h["epoch"] <= start_epoch]
					metrics_data = loaded_metrics
					print(f"[RESUME] Восстановлена история метрик ({len(metrics_data['history'])} эпох)")
				except Exception as e:
					print(f"[ОШИБКА] Не удалось прочитать {metrics_file}: {e}")

		print(f"[RESUME] Обучение продолжится с эпохи {start_epoch + 1}\n")

	# ======== Цикл обучения ========
	scaler = torch.amp.GradScaler('cuda') 
	for epoch in range(start_epoch, config_train.EPOCHS):
		model.train()
		epoch_loss = 0

		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config_train.EPOCHS}")
		for images, masks in pbar:
			images = images.to(device, non_blocking=True)
			masks = masks.to(device, non_blocking=True)
			if DEBUG:
				print("DEBUG: Батч перенесен на GPU")

			optimizer.zero_grad()
			with torch.amp.autocast('cuda'):
				outputs = model(images)
				loss = criterion(outputs, masks)
			
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			epoch_loss += loss.item()
			pbar.set_postfix(loss=loss.item())

		avg_loss = epoch_loss / len(train_loader)
		writer.add_scalar("Loss/Train", avg_loss, epoch)

		# Валидация
		model.eval()
		val_loss = 0

		keys = list(range(config_train.NUM_CLASSES)) + ["any_obstacle"]
		total_metrics = {k: {"intersection": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0} for k in keys}

		# Сохраняем первый батч для визуализации
		# P.S.: можно было бы и брать последний батч, на такой вариант
		# немного чище
		visual_images, visual_masks, visual_preds = None, None, None

		with torch.no_grad():
			pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{config_train.EPOCHS}")
			for batch_idx, (val_images, val_masks) in enumerate(pbar):
				val_images = val_images.to(device)
				val_masks = val_masks.to(device)

				# Замеряем время инференса
				start_event = torch.cuda.Event(enable_timing=True)
				end_event = torch.cuda.Event(enable_timing=True)

				start_event.record()
				val_outputs = model(val_images)

				loss = criterion(val_outputs, val_masks)
				val_loss += loss.item()

				# Метрики IOU, Precision, Recall
				batch_metrics = get_intersection_and_union(val_outputs, val_masks, config_train.NUM_CLASSES)
				for key in keys:
					for metric_name in["intersection", "union", "tp", "fp", "fn"]:
						total_metrics[key][metric_name] += batch_metrics[key][metric_name]

				# Сохраняем первый батч для визуализации
				if batch_idx == 0:
					visual_images = val_images.cpu()
					visual_masks = val_masks.cpu()
					visual_preds = torch.argmax(val_outputs, dim=1).cpu()

			avg_val_loss = val_loss / len(val_loader)

			# Считаем IoU для каждого класса за эпоху
			mean_ious = {}
			for key in keys:
				union = total_metrics[key]["union"]
				intersection = total_metrics[key]["intersection"]
				mean_ious[key] = intersection / union if union > 0 else 0.0 # Если класса не было в валидации

			# Общий mIoU
			# считаем только по базовым классам
			valid_class_ious = [mean_ious[i] for i in range(config_train.NUM_CLASSES) if total_metrics[i]["union"] > 0]
			mIoU = np.mean(valid_class_ious) if valid_class_ious else 0.0

			# Precision для Safe_Ground (Класс 0) -> Насколько мы уверены, что там безопасно
			tp_sg = total_metrics[0]["tp"]
			fp_sg = total_metrics[0]["fp"]
			safe_ground_precision = tp_sg / (tp_sg + fp_sg) if (tp_sg + fp_sg) > 0 else 0.0

			# Recall для Any_Obstacle -> Какую долю реальных препятствий мы обнаружили
			tp_obs = total_metrics["any_obstacle"]["tp"]
			fn_obs = total_metrics["any_obstacle"]["fn"]
			any_obstacle_recall = tp_obs / (tp_obs + fn_obs) if (tp_obs + fn_obs) > 0 else 0.0

			writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
			writer.add_scalar("Metrics_IoU/mIoU_All_Classes", mIoU, epoch)
			writer.add_scalar("Metrics_IoU/Safe_Ground", mean_ious[0], epoch)
			writer.add_scalar("Metrics_IoU/Static_Obstacle", mean_ious[1], epoch)
			writer.add_scalar("Metrics_IoU/Vegetation", mean_ious[2], epoch)
			writer.add_scalar("Metrics_IoU/Dynamic_Obstacle", mean_ious[3], epoch)
			writer.add_scalar("Metrics_IoU/Any_Obstacle", mean_ious["any_obstacle"], epoch)

			writer.add_scalar("Metrics_safety/Precision_Safe_Ground", safe_ground_precision, epoch)
			writer.add_scalar("Metrics_safetySafety/Recall_Any_Obstacle", any_obstacle_recall, epoch)
			
			writer.add_image("Visual/1_Image", visual_images[0], epoch)
			writer.add_image("Visual/2_True_Mask", paint_segmap(visual_masks[0]), epoch)
			writer.add_image("Visual/3_Pred_Mask", paint_segmap(visual_preds[0]), epoch)

			# Сохраняем модель каждую эпоху
			epoch_model_name = f"model_epoch_{epoch+1}.pth"
			torch.save({
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'best_val_iou': best_val_iou,
			}, os.path.join(run_dir, epoch_model_name))

			# Сохраняем лучшую модель
			if mean_ious[0] > best_val_iou:
				best_val_iou = mean_ious[0]
				torch.save({
					'epoch': epoch + 1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_iou': best_val_iou,
				}, os.path.join(run_dir, "best_model.pth"))
				print(f"\n[SAVE] Обновлена лучшая модель на эпохе {epoch+1}")

		epoch_metrics = {
			"epoch": epoch + 1,
			"train_loss": round(avg_loss, 4),
			"val_loss": round(avg_val_loss, 4),
			"mIoU": round(mIoU, 4),
			"safe_ground_iou": round(mean_ious[0], 4),
			"safe_ground_precision": round(safe_ground_precision, 4),
			"any_obstacle_recall": round(any_obstacle_recall, 4)
		}
		metrics_data["history"].append(epoch_metrics)
		with open(metrics_file, "w", encoding="utf-8") as f:
			json.dump(metrics_data, f, indent=4, ensure_ascii=False)

		print(f"Epoch[{epoch+1}/{config_train.EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {mIoU:.4f} | SafeGround_IoU: {mean_ious[0]:.4f} | Any_Obstacle_IoU: {mean_ious['any_obstacle']:.4f}")

	writer.close()

if __name__ == "__main__":
	train_model()