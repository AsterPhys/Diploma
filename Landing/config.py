# ===============================================
# Конфигурация для обучения семантической посадки
# ===============================================

import os

# --- ПУТИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RGB_DIR = os.path.join(DATA_DIR, 'rgb')
DEPTH_DIR = os.path.join(DATA_DIR, 'depth')
MASK_DIR = os.path.join(DATA_DIR, 'mask')
DEPTH_VIS_DIR = os.path.join(DATA_DIR, 'depth_vis')
MASK_VIS_DIR = os.path.join(DATA_DIR, 'mask_vis')

# --- UNREAL ENGINE SETTINGS ---
UE_EXECUTABLE = r"D:\ProgramFiles\UE_5.4\Engine\Binaries\Win64\UnrealEditor.exe"
UE_PROJECT_PATH = r"D:\UnrealProjects\Diploma\Diploma.uproject"
MAP_NAME = "DowntownWestLevel2"
UE_PROCESS_NAME = "UnrealEditor.exe"
TIME2WAIT = 20

# --- НАСТРОЙКИ СБОРА ДАННЫХ ---
DATASET_SIZE = 10_000
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
Z_MIN, Z_MAX = -0.5, -10.0

ROLL_PITCH_NOISE = 5.0  

# --- ПАРАМЕТРЫ КАМЕРЫ ---
CAMERA_NAME = "bottom_center"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
CAMERA_FOV_DEG = 90.0

# --- АУГМЕНТАЦИИ СРЕДЫ (ПОГОДА И ВРЕМЯ СУТОК) ---
ENABLE_ENV_RANDOMIZATION = True
ENV_UPDATE_FREQUENCY = 20  # менять погоду и время каждые ... кадров

# Диапазон времени суток (в часах, от 0 до 24)
# в диапазоне, чтобы совсем ночью вслепую не летать
TIME_OF_DAY_MIN = 6
TIME_OF_DAY_MAX = 18