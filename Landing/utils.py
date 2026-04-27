import os
import argparse
import config
import cv2
import numpy as np

def remove_and_shift(indices_to_delete):
    """
    Удаляет указанные кадры и сдвигает нумерацию всех последующих.
    """
    folders = {
        config.RGB_DIR: '.png',
        config.DEPTH_DIR: '.npy',
        config.MASK_DIR: '.png',
        config.DEPTH_VIS_DIR: '.png',
        config.MASK_VIS_DIR: '.png'
    }

    # Получаем список всех существующих индексов
    if not os.path.exists(config.RGB_DIR):
        print(f"Папка {config.RGB_DIR} не найдена.")
        return

    existing_files = os.listdir(config.RGB_DIR)
    existing_indices =[]
    for f in existing_files:
        if f.endswith('.png'):
            try:
                existing_indices.append(int(f.split('.')[0]))
            except ValueError:
                pass
                
    existing_indices.sort()

    if not existing_indices:
        print("Датасет пуст.")
        return

    # Удаляем указанные файлы
    deleted_count = 0
    for idx in indices_to_delete:
        if idx in existing_indices:
            for folder, ext in folders.items():
                filepath = os.path.join(folder, f"{idx:05d}{ext}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
            existing_indices.remove(idx)
            deleted_count += 1
            print(f"[-] Удален кадр: {idx:05d}")
        else:
            print(f"[!] Кадр {idx:05d} не найден, пропускаем.")

    # Переименовываем оставшиеся файлы
    print("Выравнивание индексов датасета...")
    shifted_count = 0
    
    for new_idx, old_idx in enumerate(existing_indices):
        if new_idx != old_idx:
            for folder, ext in folders.items():
                old_path = os.path.join(folder, f"{old_idx:05d}{ext}")
                new_path = os.path.join(folder, f"{new_idx:05d}{ext}")
                
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            shifted_count += 1

    print("=" * 40)
    print(f"Успешно удалено кадров: {deleted_count}")
    print(f"Сдвинуто/переименовано кадров: {shifted_count}")
    print(f"Новый размер датасета: {len(existing_indices)}")
    print(f"Последний индекс: {len(existing_indices) - 1 if existing_indices else -1}")
    print("=" * 40)

def find_invalid_masks():
	"""
	Ищет кадры, где на маске есть нераспознанные цвета.
	В collector.py нераспознанные цвета остаются нулями (0) на class_mask.
	"""
	print("[*] Поиск масок с невалидными цветами...")
	invalid_indices =[]
	
	if not os.path.exists(config.MASK_DIR):
		return invalid_indices
		
	for f in os.listdir(config.MASK_DIR):
		if f.endswith('.png'):
			idx = int(f.split('.')[0])
			mask_path = os.path.join(config.MASK_DIR, f)
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			
			# Если есть хотя бы один пиксель = 0, значит цвет не из словаря
			if np.any(mask == 0):
				invalid_indices.append(idx)
				
	print(f"Найдено бракованных масок: {len(invalid_indices)}")
	return invalid_indices


def find_dark_images(threshold=30, max_dark_ratio=0.7):
	"""
	Ищет кадры, которые более чем на max_dark_ratio слишком темные.
	Используется V-канал (HSV), чтобы отсеивать не только чисто черные кадры.
	"""
	print(f"[*] Поиск темных кадров (Порог яркости: {threshold}/255, Площадь тьмы > {max_dark_ratio*100}%)...")
	dark_indices =[]
	
	if not os.path.exists(config.RGB_DIR):
		return dark_indices
		
	for f in os.listdir(config.RGB_DIR):
		if f.endswith('.png'):
			idx = int(f.split('.')[0])
			img_path = os.path.join(config.RGB_DIR, f)
			img = cv2.imread(img_path)
			
			# Переводим BGR в HSV.
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			# Извлекаем канал value (яркость).
			v_channel = hsv[:, :, 2]
			
			# Считаем количество пикселей темнее порога.
			dark_pixels = np.sum(v_channel < threshold)
			total_pixels = v_channel.size
			dark_ratio = dark_pixels / total_pixels
			
			if dark_ratio > max_dark_ratio:
				dark_indices.append(idx)
				
	print(f"Найдено слишком темных кадров: {len(dark_indices)}")
	return dark_indices

def main():
	parser = argparse.ArgumentParser(description="Утилита для очистки датасета AirSim")
	
	# Ручное удаление
	parser.add_argument('-d', '--delete', nargs='+', type=int, 
						help='Удалить кадры вручную по номерам. Пример: -d 5 12 105.')
	
	# Автоматическая очистка масок
	parser.add_argument('--clean-masks', action='store_true', 
						help='Автоматически найти и удалить кадры с невалидными цветами на маске.')
	
	# Автоматическая очистка темных кадров
	parser.add_argument('--clean-dark', action='store_true', 
						help='Автоматически найти и удалить слишком темные кадры.')
	parser.add_argument('--dark-thresh', type=int, default=30, 
						help='Порог темноты от 0 до 255.')
	parser.add_argument('--dark-ratio', type=float, default=0.7, 
						help='Доля темных пикселей (0.0 - 1.0) для удаления.')

	args = parser.parse_args()

	indices_to_delete = set()

	if args.delete:
		indices_to_delete.update(args.delete)

	if args.clean_masks:
		indices_to_delete.update(find_invalid_masks())

	if args.clean_dark:
		indices_to_delete.update(find_dark_images(args.dark_thresh, args.dark_ratio))

	if indices_to_delete:
		print(f"\nИтого к удалению уникальных кадров: {len(indices_to_delete)}")
		remove_and_shift(list(indices_to_delete))
	else:
		# Если запущено без аргументов или ничего не найдено
		if not (args.delete or args.clean_masks or args.clean_dark):
			parser.print_help()
		else:
			print("Удалять нечего, датасет чист!")

if __name__ == "__main__":
	main()