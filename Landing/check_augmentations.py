import os
import cv2
import albumentations as A
from datetime import datetime
from tqdm import tqdm

INPUT_DIR = 'dataset\test\rgb'  # Путь к папке с картинками
OUTPUT_BASE_DIR = './augmentation_previews'
N_IMAGES = 10  # Сколько изображений обработать
IMG_HEIGHT = 480  # Твой config.IMAGE_HEIGHT
IMG_WIDTH = 640   # Твой config.IMAGE_WIDTH

def run_augmentations():
    # 1. Создаем папку с датой и временем
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(OUTPUT_BASE_DIR, f"session_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Определяем только аугментации (как в твоем классе EvalDataset)
    # В твоем коде используется CenterCrop. 
    # Normalize и ToTensor обычно не используются для сохранения "глазами", 
    # так как они портят цвета для обычного просмотра.
    aug_pipeline = A.Compose([
        A.CenterCrop(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
    ])

    # 3. Получаем список файлов
    img_names = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    img_names = img_names[:N_IMAGES]

    print(f"[*] Обработка {len(img_names)} изображений...")
    
    for i, name in enumerate(tqdm(img_names)):
        img_path = os.path.join(INPUT_DIR, name)
        image = cv2.imread(img_path)
        
        if image is None:
            continue

        # Применяем аугментацию
        augmented = aug_pipeline(image=image)
        res_img = augmented['image']

        # 4. Формируем название: оригинальное_имя + номер
        name_without_ext = os.path.splitext(name)[0]
        ext = os.path.splitext(name)[1]
        save_name = f"{name_without_ext}_{i}{ext}"
        
        cv2.imwrite(os.path.join(save_dir, save_name), res_img)

    print(f"[УСПЕХ] Изображения сохранены в: {save_dir}")

if __name__ == "__main__":
    run_augmentations()