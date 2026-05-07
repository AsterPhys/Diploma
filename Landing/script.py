import os
import shutil
import re
from PIL import Image

def repair_and_reindex(train_dir, test_dir):
    TARGET_W, TARGET_H = 640, 480
    
    subfolders = {
        'rgb': '.png',
        'depth': '.npy',
        'mask': '.png',
        'depth_vis': '.png',
        'mask_vis': '.png'
    }
    
    train_rgb_dir = os.path.join(train_dir, 'rgb')
    test_rgb_dir = os.path.join(test_dir, 'rgb')
    
    if not os.path.exists(train_rgb_dir) or not os.path.exists(test_rgb_dir):
        print("Ошибка: Не найдены папки 'rgb' в train или test. Проверь пути!")
        return

    # ==========================================
    # ШАГ 1: Чистим TEST от битых размеров
    # ==========================================
    print("1. Сканируем папку TEST и удаляем брак...")
    test_files = sorted([f for f in os.listdir(test_rgb_dir) if f.endswith('.png')])
    
    if not test_files:
        print("Папка test пуста!")
        return
        
    # Умное определение формата (вытаскиваем нули и длину по первому файлу)
    first_test_base = os.path.splitext(test_files[0])[0]
    match_start = re.match(r"^(.*?)(\d+)$", first_test_base)
    start_index = int(match_start.group(2)) if match_start else 0
    prefix = match_start.group(1) if match_start else ""
    num_length = len(match_start.group(2)) if match_start else 5

    good_test_bases =[]
    
    for test_file in test_files:
        test_base = os.path.splitext(test_file)[0]
        test_rgb_path = os.path.join(test_rgb_dir, test_file)
        
        try:
            with Image.open(test_rgb_path) as img:
                w, h = img.size
        except Exception:
            continue
            
        if w != TARGET_W or h != TARGET_H:
            print(f"  [-] Удален битый сэмпл из test: {test_base} ({w}x{h})")
            for sub, ext in subfolders.items():
                p = os.path.join(test_dir, sub, test_base + ext)
                if os.path.exists(p):
                    os.remove(p)
        else:
            good_test_bases.append(test_base)

    # ==========================================
    # ШАГ 2: Чистим TRAIN и забираем доноров из TEST
    # ==========================================
    print("\n2. Сканируем папку TRAIN и чиним с помощью TEST...")
    train_files = sorted([f for f in os.listdir(train_rgb_dir) if f.endswith('.png')])
    fixed_count = 0
    
    for train_file in train_files:
        train_base = os.path.splitext(train_file)[0]
        train_rgb_path = os.path.join(train_rgb_dir, train_file)
        
        try:
            with Image.open(train_rgb_path) as img:
                w, h = img.size
        except Exception:
            continue
            
        if w != TARGET_W or h != TARGET_H:
            print(f"  [!] Найден битый сэмпл в train: {train_base} ({w}x{h}).")
            
            if not good_test_bases:
                print("ОШИБКА: В test закончились целые файлы для замены!")
                return
            
            # Берем самый последний хороший файл из test
            replacement_base = good_test_bases.pop() 
            print(f"      -> Заменяем его связкой {replacement_base} из test.")
            
            for sub, ext in subfolders.items():
                train_p = os.path.join(train_dir, sub, train_base + ext)
                test_p = os.path.join(test_dir, sub, replacement_base + ext)
                
                # 1. Удаляем битый в train
                if os.path.exists(train_p):
                    os.remove(train_p)
                # 2. Перемещаем донора и сразу переименовываем
                if os.path.exists(test_p):
                    shutil.move(test_p, train_p)
                    
            fixed_count += 1

    # ==========================================
    # ШАГ 3: Сшиваем дыры в нумерации TEST
    # ==========================================
    print("\n3. Перенумеровываем оставшиеся файлы в TEST (убираем дыры)...")
    renamed_count = 0
    
    for i, old_base in enumerate(good_test_bases):
        # Генерируем новое имя (например: i=0 -> 00000)
        new_index = start_index + i
        new_base = f"{prefix}{new_index:0{num_length}d}"
        
        if old_base != new_base:
            for sub, ext in subfolders.items():
                old_p = os.path.join(test_dir, sub, old_base + ext)
                new_p = os.path.join(test_dir, sub, new_base + ext)
                
                # Переименовываем
                if os.path.exists(old_p):
                    os.rename(old_p, new_p)
            renamed_count += 1
            
    print(f"\n✅ ГОТОВО!")
    print(f"   Исправлено файлов в train: {fixed_count}")
    print(f"   Переименовано связок в test: {renamed_count}")

# ==========================================
# ПОДСТАВЬ СВОИ ПУТИ:
# ==========================================
TRAIN_DIR = r"dataset\train"
TEST_DIR  = r"dataset\test"

repair_and_reindex(TRAIN_DIR, TEST_DIR)