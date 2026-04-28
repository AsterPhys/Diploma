import os
import numpy as np
import albumentations as A
from config import BASE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT

MODELS_DIR = os.path.join(BASE_DIR, 'runs', 'segmentation')

# --- СИСТЕМНЫЕ НАСТРОЙКИ ---
NUM_WORKERS = 0

# --- ПАРАМЕТРЫ ТЕКУЩЕГО ЭКСПЕРИМЕНТА ---
SEG_MODEL_NAME = "UNet"    # "UNet", "DeepLabV3", "SegFormer"
SEG_BACKBONE = "resnet34"  # "resnet34", "mobilenet_v2", "mit_b0"
SEG_WEIGHTS = "imagenet"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 50
OPTIMIZER = "Adam"
CRITERION = "CrossEntropyLoss"

# --- НАСТРОЙКИ СОХРАНЕНИЯ И ЛОГИРОВАНИЯ ---
N_SAVE_EXAMPLES = 8
N_SAVE_MODEL = 5

# --- НАСТРОЙКИ КЛАССОВ ---
NUM_CLASSES = 4
COLOR_MAP = np.array([
	[240, 240, 240],   # 0: Safe_Ground
	[45, 52, 54],      # 1: Obstacles (Static + Hazard)
	[32, 201, 151],    # 2: Vegetation
	[255, 82, 82]      # 3: Dynamic_Obstacle
], dtype=np.uint8)

# --- АУГМЕНТАЦИИ ---
TRAIN_TRANSFORMS = A.Compose([
	A.RandomCrop(height=480, width=640, p=1.0),

	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
	A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
	A.GaussianBlur(blur_limit=(3, 5), p=0.2),

	A.Normalize(),
	A.pytorch.ToTensorV2()
])

VAL_TRANSFORMS = A.Compose([
	A.CenterCrop(height=480, width=640, p=1.0),
	A.Normalize(),
	A.pytorch.ToTensorV2()
])

def get_train_config_dict():
	"""Собирает все актуальные настройки в словарь для сохранения в JSON"""
	return {
		"model": {
			"architecture": SEG_MODEL_NAME,
			"backbone": SEG_BACKBONE,
			"weights": SEG_WEIGHTS,
			"num_classes": NUM_CLASSES,
		},
		"training": {
			"learning_rate": LEARNING_RATE,
			"batch_size": BATCH_SIZE,
			"epochs": EPOCHS,
			"optimizer": OPTIMIZER,
			"loss": CRITERION
		},
		"system": {
			"num_workers": NUM_WORKERS,
			"n_save_examples": N_SAVE_EXAMPLES,
			"n_save_model_every": N_SAVE_MODEL
		},
		"data": {
			"image_width": IMAGE_WIDTH,
			"image_height": IMAGE_HEIGHT,
		},
		"color_map": COLOR_MAP.tolist(), 
		"augmentations": A.to_dict(TRAIN_TRANSFORMS) 
	}