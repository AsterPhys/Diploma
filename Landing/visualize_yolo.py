import os
import cv2
import numpy as np
import random
import config

YOLO_DIR = config.YOLO_DATA_DIR
VIS_DIR = config.YOLO_VISUALIZATION_DIR
NUM_SAMPLES = 20

COLORS = {
    0: (200, 200, 200),  # Safe_Ground
    1: (45, 52, 54),     # Static_Obstacle
    2: (32, 201, 151),   # Vegetation
    3: (82, 82, 255)     # Dynamic_Obstacle
}

CLASS_NAMES = {
    0: 'Safe_Ground',
    1: 'Static_Obstacle',
    2: 'Vegetation',
    3: 'Dynamic_Obstacle'
}

def main():
    print(f"[*] Подготовка папки для визуализации: {VIS_DIR}")
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # Берем картинки из папки train
    img_dir = os.path.join(YOLO_DIR, 'images', 'train')
    lbl_dir = os.path.join(YOLO_DIR, 'labels', 'train')
    
    if not os.path.exists(img_dir):
        print(f"[!] Ошибка: Папка {img_dir} не найдена. Запустите сначала prepare_yolo_dataset.py")
        return

    images =[f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if not images:
        print("[!] Папка с картинками пуста.")
        return

    # Выбираем N случайных картинок
    samples = random.sample(images, min(NUM_SAMPLES, len(images)))
    
    for img_name in samples:
        img_path = os.path.join(img_dir, img_name)
        txt_name = img_name.rsplit('.', 1)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, txt_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        overlay = img.copy()
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                    
                class_id = int(parts[0])
                color = COLORS.get(class_id, (0, 255, 0)) # Зеленый цвет по умолчанию, если id не найден
                
                # Извлекаем координаты X, Y и переводим из нормализованных (0-1) в пиксели
                coords = [float(p) for p in parts[1:]]
                pts = np.array(coords).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts = pts.astype(np.int32)
                
                cv2.fillPoly(overlay, [pts], color)
                # Рисуем четкую границу контура на основном изображении
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
                
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        save_path = os.path.join(VIS_DIR, img_name)
        cv2.imwrite(save_path, img)
        
    print(f"[*] Готово! Сохранено {len(samples)} примеров разметки в папку: {VIS_DIR}")

if __name__ == "__main__":
    main()