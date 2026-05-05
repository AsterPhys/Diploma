import os
import json
import numpy as np
import albumentations as A
import cv2
from config import BASE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT

MODELS_DIR = os.path.join(BASE_DIR, 'runs', 'segmentation')

# --- СИСТЕМНЫЕ НАСТРОЙКИ ---
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

# --- ПАРАМЕТРЫ ТЕКУЩЕГО ЭКСПЕРИМЕНТА ---
SEG_MODEL_NAME = os.environ.get("SEG_MODEL_NAME", "Unet")
SEG_BACKBONE = os.environ.get("SEG_BACKBONE", "resnet34")
SEG_WEIGHTS = os.environ.get("SEG_WEIGHTS", "imagenet")
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
EPOCHS = int(os.environ.get("EPOCHS", 50))
OPTIMIZER = os.environ.get("OPTIMIZER", "Adam")
CRITERION = os.environ.get("CRITERION", "CrossEntropyLoss")

EXTRA_KWARGS = json.loads(os.environ.get("EXTRA_KWARGS", "{}"))

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
# Получаем стратегию из окружения
SELECTED_AUG = os.environ.get("AUG_STRATEGY", "spatial")

# Пресеты аугментаций (кумулятивный подход)
AUG_STRATEGIES = {
	# Только нормализация
    "none": A.Compose([
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ]),

	# Геометрические
	"spatial": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15), 
            translate_percent=(-0.1, 0.1), 
            rotate=(-45, 45), 
            border_mode=cv2.BORDER_REFLECT_101, 
            p=0.5
        ),
        
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ]),

	# Геометрия + свет
	"lighting": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15), 
            translate_percent=(-0.1, 0.1), 
            rotate=(-45, 45), 
            border_mode=cv2.BORDER_REFLECT_101, 
            p=0.5
        ),
        
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ]),

	# Геометрия + свет + деградация камеры
	"sensor": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15), 
            translate_percent=(-0.1, 0.1), 
            rotate=(-45, 45), 
            border_mode=cv2.BORDER_REFLECT_101, 
            p=0.5
        ),
        
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(std_range=(5.0, 15.0), p=0.4),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ])
}

TRAIN_TRANSFORMS = AUG_STRATEGIES.get(SELECTED_AUG, AUG_STRATEGIES["spatial"])

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
		"extra_kwargs": EXTRA_KWARGS,
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
		"augmentations": {
			"strategy": SELECTED_AUG,
			"train_set": A.to_dict(TRAIN_TRANSFORMS) ,
			"val_set": A.to_dict(VAL_TRANSFORMS) 
		}
	}