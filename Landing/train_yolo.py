import os

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_USER_DB_PATH'] = 'D:\\Cache\\Temp'
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = 'D:\\Cache\\Temp'

from ultralytics import YOLO

def main():
    print("[YOLO] Загрузка архитектуры YOLO11n-seg...")
    model = YOLO("yolo11n-seg.pt")

    print("[YOLO] Старт обучения...")
    results = model.train(
        data="drone_yolo.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        project="runs/yolo",
        name="yolo11n_drone_run",
        
        # --- АУГМЕНТАЦИИ ---
        hsv_h=0.015,        # Цветовой джиттер (Hue)
        hsv_s=0.5,          # Насыщенность (Saturation)
        hsv_v=0.4,          # Яркость (Value)
        degrees=15.0,       # Повороты дрона
        translate=0.1,      # Сдвиги камеры
        scale=0.3,          # Зум
        flipud=0.5,         # Отражение по вертикали 50%
        fliplr=0.5,         # Отражение по горизонтали 50%
        mosaic=1.0,         # Мозаика (склейка 4 фото в 1)
        erasing=0.2,        # Random Erasing
        
        optimizer="AdamW",
        lr0=1e-3
    )
    
    print("[YOLO] Обучение завершено.")

if __name__ == "__main__":
    main()