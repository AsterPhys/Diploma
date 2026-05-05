import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
import config

YOLO_DIR = config.YOLO_DATA_DIR
VAL_SPLIT = 0.2

def create_dirs():
	for split in ['train', 'val']:
		os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
		os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

def process_mask(mask_path, txt_path, width, height):
	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	if mask is None: return False

	# Перемаппинг классов
	new_mask = np.zeros_like(mask, dtype=np.int32)
	new_mask[mask == 5] = 0                    # Safe_Ground
	new_mask[(mask == 1) | (mask == 3)] = 1    # Static_Obstacle
	new_mask[mask == 4] = 2                    # Vegetation
	new_mask[mask == 2] = 3                    # Dynamic_Obstacle

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

	# Поиск полигонов
	lines =[]
	for class_id in range(4):
		# Бинаризуем маску для конкретного класса
		class_bin = np.uint8(new_mask == class_id) * 255
		
		# Немного видоизменяем маски для yolo:
		# склеиваем объекты в более цельные пятна
		class_bin = cv2.morphologyEx(class_bin, cv2.MORPH_CLOSE, kernel)

		# Находим контуры
		contours, _ = cv2.findContours(class_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		for cnt in contours:
			# Упрощаем контур
			epsilon = 0.001 * cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, epsilon, True)
			
			# Для полигона нужно минимум 3 точки
			if len(approx) < 3:
				continue
				
			# Нормализуем координаты
			coords = []
			for point in approx:
				x, y = point[0]
				coords.append(f"{x / width:.5f}")
				coords.append(f"{y / height:.5f}")
				
			lines.append(f"{class_id} " + " ".join(coords))
			
	with open(txt_path, 'w') as f:
		f.write("\n".join(lines))
	return True

def process_split(img_list, split_name):
	for img_name in tqdm(img_list, desc=f"Конвертация {split_name}"):
		src_img = os.path.join(config.RGB_DIR, img_name)
		src_mask = os.path.join(config.MASK_DIR, img_name)
		
		img = cv2.imread(src_img)
		if img is None:
			continue
			
		h, w, _ = img.shape
		
		txt_name = img_name.replace('.png', '.txt')
		dst_img = os.path.join(YOLO_DIR, 'images', split_name, img_name)
		dst_txt = os.path.join(YOLO_DIR, 'labels', split_name, txt_name)
		
		if process_mask(src_mask, dst_txt, w, h):
			shutil.copy(src_img, dst_img)

def main():
	print("[YOLO] Создание структуры папок...")
	create_dirs()
	
	images = [f for f in os.listdir(config.RGB_DIR) if f.endswith('.png')]
	images.sort()
	random.shuffle(images)
	
	val_size = int(len(images) * VAL_SPLIT)
	val_images = images[:val_size]
	train_images = images[val_size:]

	print(f"\n[YOLO] Всего картинок: {len(images)}")
	print(f"[YOLO] Пойдет в Train: {len(train_images)}")
	print(f"[YOLO] Пойдет в Val:   {len(val_images)}\n")

	process_split(train_images, 'train')
	process_split(val_images, 'val')

	print(f"\n[YOLO] Готово! Датасет сохранен в: {YOLO_DIR}")

if __name__ == "__main__":
	main()