import airsim
import numpy as np

# Твои классы и их Custom Stencil ID
classes = {
    1: "Static_Obstacle",
    2: "Dynamic_Obstacle",
    3: "Hazard",
    4: "Vegetation",
    5: "Safe_Ground"
}

client = airsim.MultirotorClient()
client.confirmConnection()

# Ставим игру на паузу для быстрых вычислений
client.simPause(True)
print("Начинаем сканирование цветов маски...")

for stencil_id, class_name in classes.items():
    # 1. Красим абсолютно ВСЕ меши на уровне в текущий ID (regex ".*" означает "всё")
    client.simSetSegmentationObjectID(".*", stencil_id, True)
    
    # 2. Даем движку 4 кадра на обновление Custom Depth буфера
    client.simContinueForFrames(4)
    
    # 3. Делаем снимок с твоей камеры
    responses = client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
    ])
    
    mask1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_mask_rgb = mask1d.reshape(responses[0].height, responses[0].width, 3)
    
    # 4. Ищем уникальные цвета в кадре (убираем черный фон[0, 0, 0])
    unique_colors = np.unique(img_mask_rgb.reshape(-1, 3), axis=0)
    colors = [c for c in unique_colors.tolist() if c !=[0, 0, 0]]
    
    # Так как мы покрасили всю карту, цвет будет всего один
    if colors:
        print(f"ID {stencil_id} ({class_name}) -> RGB: {colors[0]}")
    else:
        print(f"ID {stencil_id} ({class_name}) -> Цвет не найден (камера смотрит в пустоту?)")

client.simPause(False)
print("\nГотово! Нажми Stop и снова Play в Unreal Engine, чтобы вернуть твои оригинальные цвета объектов.")