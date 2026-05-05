# ===============================================
# Конфигурация для обучения семантической посадки
# ===============================================

import os
import numpy as np

# --- ПУТИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
DATA2DEL_DIR = os.path.join(BASE_DIR, 'data_deleted')

RGB_DIR = os.path.join(DATA_DIR, 'rgb')
DEPTH_DIR = os.path.join(DATA_DIR, 'depth')
MASK_DIR = os.path.join(DATA_DIR, 'mask')
DEPTH_VIS_DIR = os.path.join(DATA_DIR, 'depth_vis')
MASK_VIS_DIR = os.path.join(DATA_DIR, 'mask_vis')

MODELS_DIR = os.path.join(BASE_DIR, 'runs', 'segmentation')

YOLO_DATA_DIR = os.path.join(BASE_DIR, 'yolo_train_data')
YOLO_VISUALIZATION_DIR = os.path.join(BASE_DIR, 'yolo_vis')

# --- UNREAL ENGINE SETTINGS ---
UE_EXECUTABLE = r"D:\ProgramFiles\UE_5.4\Engine\Binaries\Win64\UnrealEditor.exe"
UE_PROJECT_PATH = r"D:\UnrealProjects\Diploma\Diploma.uproject"
MAP_NAME = "DowntownWestLevel2"
UE_PROCESS_NAME = "UnrealEditor.exe"
TIME2WAIT = 20

# --- ПАРАМЕТРЫ СЕТКИ (GRID SPAWN) ---
GRID_STEP = 5.0     # Шаг сетки в метрах
GRID_JITTER = 2.0   # Случайное отклонение от узла сетки

# --- СБОР ДАННЫХ ---
DATASET_SIZE = 15_000
# Отступ от стен полигона при спавне (в метрах)
# Гарантирует, что дрон не появится вплотную к стене
SAFE_MARGIN = 2.0

# Лимиты для телепортации дрона (границы наполненной карты)
SPAWN_POLYGON =[
    (-287.2, 93.9),
	(-287.25, -120.95),
	(98.35, -120.95),
	(238.5, -3.1),
	(128, 90.05),
	(127.55, 236.65),
	(-39.25, 236.85),
]
# Лимиты зоны, в которой дрон будет спавниться чаще
HOTZONE_POLYGON = [
    (-253.45, -68.65),
	(-255.6, 75.85),
	(128.6, 77.1),
	(130.15, -74.25),
]
# Вероятность спавна в Hotzone
HOTZONE_PROBABILITY = 0.7

Z_MIN, Z_MAX = -0.5 + 1, -30.0 + 1 # учтено, что PlayerStart на высоте 1 м
ROLL_PITCH_NOISE = 5.0  

# --- ПАРАМЕТРЫ КАМЕРЫ ---
CAMERA_NAME = "bottom_center"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FOV_DEG = 90.0

# --- АУГМЕНТАЦИИ СРЕДЫ (ПОГОДА И ВРЕМЯ СУТОК) ---
ENABLE_ENV_RANDOMIZATION = True
ENV_UPDATE_FREQUENCY = 50  # менять погоду и время каждые ... кадров

# Диапазон времени суток
TIME_OF_DAY_MIN = 5
TIME_OF_DAY_MAX = 21