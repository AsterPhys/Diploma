import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm

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

os.environ['HSA_XNACK'] = '0'

DEBUG = True

def paint_segmap(mask):
	"""Получает из маски сегментации с классами маску с цветами."""
	if torch.is_tensor(mask):
		mask = mask.cpu().numpy()
	rgb = config_train.COLOR_MAP[mask]
	# Транспонируем для TensorBoard (HWC -> CHW).
	return np.transpose(rgb, (2, 0, 1))

def calculate_iou(pred_masks, true_masks, num_classes):
	"""
	Считает IoU для каждого батча. 
	Возвращает словарь {class_id: iou}
	"""
	# pred_masks имеют форму [B, C, H, W] (логиты)
	# Получаем предсказанный класс, за счет чего получаем
	# размерность [B, H, W]
	preds = torch.argmax(pred_masks, dim=1)

	ious = {}
	# IoU для каждого класса
	for cls in range(num_classes):
		pred_inds = (preds == cls)
		target_inds = (true_masks == cls)

		intersection = (pred_inds & target_inds).sum().float()
		union = (pred_inds | target_inds).sum().float()

		# Если этого класса вообще нет ни в таргете, ни в предсказании,
		# мы его пропускаем.
		if union > 0:
			ious[cls] = (intersection / union).item()
		else:
			ious[cls] = float('nan')

	# Бинарный IoU (Safe_Ground vs Все остальные классы - препятствия)
	pred_danger = (preds > 0)
	target_danger = (true_masks > 0)

	danger_intersection = (pred_danger & target_danger).sum().float()
	danger_union = (pred_danger | target_danger).sum().float()

	if danger_union > 0:
		ious["any_obstacle"] = (danger_intersection / danger_union).item()
	else:
		ious["any_obstacle"] = float('nan')

	return ious

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

		# HWC -> CHW + нормализация
		image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

		return torch.tensor(image), torch.tensor(new_mask, dtype=torch.long)

def train_model():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.backends.cudnn.benchmark = True
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
	if config_train.SEG_MODEL_NAME == "UNet":
		model = smp.Unet(
			encoder_name=config_train.SEG_BACKBONE,
			encoder_weights=config_train.SEG_WEIGHTS,
			in_channels=3,
			classes=config_train.NUM_CLASSES
		).to(device)

	if config_train.OPTIMIZER == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=config_train.LEARNING_RATE)
	
	if config_train.CRITERION == "CrossEntropyLoss":
		criterion = nn.CrossEntropyLoss()

	# ======== Цикл обучения ========
	best_val_iou = 0.0

	for epoch in range(config_train.EPOCHS):
		model.train()
		epoch_loss = 0

		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config_train.EPOCHS}")
		for images, masks in pbar:
			images = images.to(device, non_blocking=True)
			masks = masks.to(device, non_blocking=True)
			if DEBUG:
				print("DEBUG: Батч перенесен на GPU")

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			pbar.set_postfix(loss=loss.item())

		avg_loss = epoch_loss / len(train_loader)
		writer.add_scalar("Loss/Train", avg_loss, epoch)

		# Валидация
		model.eval()
		val_loss = 0

		class_ious = {i: [] for i in range(config_train.NUM_CLASSES)}
		class_ious["any_obstacle"] = []

		# Сохраняем первый батч для визуализации
		# P.S.: можно было бы и брать последний батч, на такой вариант
		# немного чище
		visual_images, visual_masks, visual_preds = None, None, None

		with torch.no_grad():
			pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{config_train.EPOCHS}")
			for batch_idx, (val_images, val_masks) in enumerate(pbar):
				val_images = val_images.to(device)
				val_masks = val_masks.to(device)

				val_outputs = model(val_images)

				loss = criterion(val_outputs, val_masks)
				val_loss += loss.item()

				# Метрика IOU
				batch_ious = calculate_iou(val_outputs, val_masks, config_train.NUM_CLASSES)
				for cls, iou in batch_ious.items():
					if not np.isnan(iou):
						class_ious[cls].append(iou)

				# Сохраняем первый батч для визуализации
				if batch_idx == 0:
					visual_images = val_images.cpu()
					visual_masks = val_masks.cpu()
					visual_preds = torch.argmax(val_outputs, dim=1).cpu()

			avg_val_loss = val_loss / len(val_loader)

			# Считаем средний IoU для каждого класса
			mean_ious = {}
			for cls in range(config_train.NUM_CLASSES):
				if len(class_ious[cls]) > 0:
					mean_ious[cls] = np.mean(class_ious[cls])
				else:
					mean_ious[cls] = 0.0
				
			# Общий mIoU
			# считаем только по базовым 4 классам
			mIoU = np.mean([mean_ious[i] for i in range(config_train.NUM_CLASSES)])
			
			writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
			writer.add_scalar("Metrics/mIoU_All_Classes", mIoU, epoch)
			# Отдельно сохраняем IoU для Safe_Ground
			writer.add_scalar("Metrics/IoU_Safe_Ground", mean_ious[0], epoch)
			writer.add_scalar("Metrics/IoU_Obstacles", mean_ious[1], epoch)
			writer.add_scalar("Metrics/IoU_Any_Obstacle", mean_ious["any_obstacle"], epoch)

			writer.add_image("Visual/1_Image", visual_images[0], epoch)
			writer.add_image("Visual/2_True_Mask", paint_segmap(visual_masks[0]), epoch)
			writer.add_image("Visual/3_Pred_Mask", paint_segmap(visual_preds[0]), epoch)

			# Сохраняем модель каждую эпоху
			epoch_model_name = f"model_epoch_{epoch+1}.pth"
			torch.save(model.state_dict(), os.path.join(run_dir, epoch_model_name))

			# Сохраняем лучшую модель
			if mean_ious[0] > best_val_iou:
				best_val_iou = mean_ious[0]
				torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
				print(f"\n[SAVE] Обновлена лучшая модель на эпохе {epoch+1}")

		print(f"Epoch [{epoch+1}/{config_train.EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {mIoU:.4f} | SafeGround_IoU: {mean_ious[0]:.4f} | Any_Obstacle_IoU: {np.mean(class_ious['any_obstacle']):.4f}")

	writer.close()

if __name__ == "__main__":
	train_model()