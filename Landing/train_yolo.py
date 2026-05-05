import os
import config
import torch

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_USER_DB_PATH'] = 'D:\\Cache\\Temp'
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = 'D:\\Cache\\Temp'

from ultralytics import YOLO

# Пресеты аугментаций
AUG_STRATEGIES = {
    # Без искажений
    "none": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "flipud": 0.0, "fliplr": 0.0, "mosaic": 0.0, "erasing": 0.0, "mixup": 0.0
    },

    # Геометрия
    "spatial": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 45.0, "translate": 0.1, "scale": 0.15,
        "flipud": 0.5, "fliplr": 0.5, "mosaic": 0.0, "erasing": 0.0, "mixup": 0.0
    },

    # Геометрия + свет
    "lighting": {
        "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.4,
        "degrees": 45.0, "translate": 0.1, "scale": 0.15,
        "flipud": 0.5, "fliplr": 0.5, "mosaic": 0.0, "erasing": 0.0, "mixup": 0.0
    },

    # Специфичные методы YOLO
    "yolo_advanced": {
        "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.4,
        "degrees": 45.0, "translate": 0.1, "scale": 0.15,
        "flipud": 0.5, "fliplr": 0.5,
        "mosaic": 1.0,    # Склеивает 4 кадра в 1
        "erasing": 0.4,   # Случайно замазывает куски кадра (имитация преград/облаков)
        "mixup": 0.1      # Смешивает 2 кадра полупрозрачно (снижает уверенность сети)
    }
}

def main():
	model_name = os.environ.get("SEG_MODEL_NAME", "yolo11n-seg.pt")
	lr = float(os.environ.get("LEARNING_RATE", 1e-3))
	batch_size = int(os.environ.get("BATCH_SIZE", 16))
	epochs = int(os.environ.get("EPOCHS", 50))
	optimizer = os.environ.get("OPTIMIZER", "auto")
	run_name = os.environ.get("RUN_NAME", "yolo_manual_run")

	project_dir = "runs/segmentation"

	selected_aug = os.environ.get("AUG_STRATEGY", "medium")
	aug_kwargs = AUG_STRATEGIES.get(selected_aug, AUG_STRATEGIES["medium"])

	print(f"\n{'='*50}")
	print("[YOLO] Загрузка архитектуры YOLO11n-seg...")
	print(f"Модель: {model_name} | LR: {lr} | BS: {batch_size} | Aug: {selected_aug}")
	print(f"{'='*50}\n")
	
	model = YOLO(model_name)

	device_id = "0" if torch.cuda.is_available() else "cpu"

	print("[YOLO] Старт обучения...")
	results = model.train(
		data="drone_yolo.yaml",
        epochs=epochs,
        imgsz=[config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
		rect=True,
        batch=batch_size,
        device=device_id,
        project=project_dir,
        name=run_name,
        optimizer=optimizer,
        lr0=lr,
		**aug_kwargs
	)
	
	print("[YOLO] Обучение {run_name} успешно завершено.")

if __name__ == "__main__":
	main()