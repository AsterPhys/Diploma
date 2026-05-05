import os
import argparse
import config
import cv2
import numpy as np
import shutil
import random

def remove_and_shift(indices_to_delete, move_files=True):
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
	
	if move_files:
		if not hasattr(config, 'DATA2DEL_DIR'):
			print("Ошибка: DATA2DEL_DIR не указан в config.py")
			return
		
		if not os.path.exists(config.DATA2DEL_DIR):
			os.makedirs(config.DATA2DEL_DIR)

	# Удаляем указанные файлы
	deleted_count = 0
	for idx in indices_to_delete:
		if idx in existing_indices:
			for folder, ext in folders.items():
				filepath = os.path.join(folder, f"{idx:05d}{ext}")
				if os.path.exists(filepath):
					if move_files:
						folder_name = os.path.basename(os.path.normpath(folder))
						target_subfolder = os.path.join(config.DATA2DEL_DIR, folder_name)
						os.makedirs(target_subfolder, exist_ok=True)
						
						base_name = f"{idx:05d}"
						dest_path = os.path.join(target_subfolder, f"{base_name}{ext}")
						
						# Проверка на совпадение имен
						counter = 1
						while os.path.exists(dest_path):
							dest_path = os.path.join(target_subfolder, f"{base_name}_{counter}{ext}")
							counter += 1
							
						shutil.move(filepath, dest_path)
					else:
						os.remove(filepath)
					
			existing_indices.remove(idx)
			deleted_count += 1
			print(f"[-] {'Перенесен' if move_files else 'Удален'} кадр: {idx:05d}")
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

def split_dataset(src_dir, dst_dir, test_count):
	"""
	Разделяет датасет на train и test, переносит файлы и переименовывает их с префиксами.
	"""
	print(f"\n[*] Разделение датасета из '{src_dir}' в '{dst_dir}'...")
	
	# Извлекаем названия подпапок из конфига
	subfolders = {
		os.path.basename(os.path.normpath(config.RGB_DIR)): '.png',
		os.path.basename(os.path.normpath(config.DEPTH_DIR)): '.npy',
		os.path.basename(os.path.normpath(config.MASK_DIR)): '.png',
		os.path.basename(os.path.normpath(config.DEPTH_VIS_DIR)): '.png',
		os.path.basename(os.path.normpath(config.MASK_VIS_DIR)): '.png'
	}

	rgb_folder = os.path.basename(os.path.normpath(config.RGB_DIR))
	rgb_src_path = os.path.join(src_dir, rgb_folder)
	
	if not os.path.exists(rgb_src_path):
		print(f"[!] Исходная папка {rgb_src_path} не найдена. Проверьте путь --src.")
		return

	# Ищем все кадры в папке src
	existing_files = os.listdir(rgb_src_path)
	existing_indices =[]
	for f in existing_files:
		if f.endswith('.png'):
			try:
				name = f.split('.')[0]
				existing_indices.append(int(name))
			except ValueError:
				pass
				
	existing_indices.sort()
	total_files = len(existing_indices)
	
	if total_files == 0:
		print("Исходный датасет пуст.")
		return
		
	if test_count >= total_files:
		print(f"Ошибка: файлов для теста ({test_count}) больше или равно общему количеству ({total_files}).")
		return

	# Рандомно выбираем кадры для теста
	test_indices = set(random.sample(existing_indices, test_count))
	train_indices = set(existing_indices) - test_indices

	for split_name in ['train', 'test']:
		for folder_name in subfolders.keys():
			os.makedirs(os.path.join(dst_dir, split_name, folder_name), exist_ok=True)

	def move_split(indices, split_name):
		count = 0
		# Сортируем индексы, чтобы новая нумерация была по порядку
		for new_idx, old_idx in enumerate(sorted(indices)):
			for folder_name, ext in subfolders.items():
				old_path = os.path.join(src_dir, folder_name, f"{old_idx:05d}{ext}")
				new_path = os.path.join(dst_dir, split_name, folder_name, f"{split_name}_{new_idx:05d}{ext}")
				
				if os.path.exists(old_path):
					shutil.move(old_path, new_path)
			count += 1
		return count

	print(f"Перенос в тест ({test_count} шт.)...")
	test_moved = move_split(test_indices, 'test')
	
	print(f"Перенос в трейн ({len(train_indices)} шт.)...")
	train_moved = move_split(train_indices, 'train')
	
	print("=" * 40)
	print("Успешно разделено!")
	print(f"Train: {train_moved} кадров -> {os.path.join(dst_dir, 'train')}")
	print(f"Test:  {test_moved} кадров -> {os.path.join(dst_dir, 'test')}")
	print("=" * 40)

def main():
	parser = argparse.ArgumentParser(description="Утилита для очистки датасета AirSim")
	
	# Ручное удаление
	parser.add_argument('--names', nargs='+', type=int, 
						help='Удалить/перенести кадры вручную по номерам. Пример: --names 5 12 105.')
	
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
	parser.add_argument('--delete', action='store_true', 
						help='Удалять кадры в папку вместо переноса.')

	parser.add_argument('--split', action='store_true', 
						help='Запустить разделение датасета на train/test.')
	parser.add_argument('--src', type=str, 
						help='Папка с исходным датасетом (которую делим).')
	parser.add_argument('--dst', type=str, 
						help='Новая папка, куда сохранятся подпапки train и test.')
	parser.add_argument('--test-count', type=int, 
						help='Количество кадров, которые пойдут в тест.')

	args = parser.parse_args()

	if args.split:
		if not args.src or not args.dst or args.test_count is None:
			print("[!] Ошибка: Для разделения обязательно укажите --src, --dst и --test-count.")
			print("Пример: python utils.py --split --src ./data --dst ./dataset_split --test-count 50")
		else:
			split_dataset(args.src, args.dst, args.test_count)
		return  # Завершаем скрипт, чтобы не выполнять очистку одновременно с разделением

	indices_to_delete = set()

	if args.names:
		indices_to_delete.update(args.names)

	if args.clean_masks:
		indices_to_delete.update(find_invalid_masks())

	if args.clean_dark:
		indices_to_delete.update(find_dark_images(args.dark_thresh, args.dark_ratio))

	if indices_to_delete:
		print(f"\nИтого к удалению уникальных кадров: {len(indices_to_delete)}")
		remove_and_shift(list(indices_to_delete), move_files=(not args.delete))
	else:
		# Если запущено без аргументов или ничего не найдено
		if not (args.names or args.clean_masks or args.clean_dark):
			parser.print_help()
		else:
			print("Удалять нечего, датасет чист!")

if __name__ == "__main__":
	main()