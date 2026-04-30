import os
from dotenv import load_dotenv

# --- ПАРАМЕТРЫ ИСПОЛЬЗУЕМЫХ МОДЕЛЕЙ --- 
RL_MODEL_NAME = "PPO_drone_29_04_2026_08_17_19"
SEG_MODEL_NAME = "UNet_resnet34_28_04_2026_23_21_55"
SEG_MODEL_EPOCH = 1

# --- ПУТИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)
HF_TOKEN = os.getenv("HF_TOKEN")

ROOT_DIR = os.path.dirname(BASE_DIR)

RL_MODEL_PATH = os.path.join(ROOT_DIR, 'RL_agent', 'models', RL_MODEL_NAME, 'latest_model.zip')

SEG_RUN_DIR = os.path.join(ROOT_DIR, 'Landing', 'runs', 'segmentation', SEG_MODEL_NAME)
SEG_MODEL_PATH = os.path.join(SEG_RUN_DIR, f'model_epoch_{SEG_MODEL_EPOCH}.pth')
SEG_CONFIG_PATH = os.path.join(SEG_RUN_DIR, 'config.json')

# --- СЕРВЕР VISION ---
VISION_SERVER_URL = "http://127.0.0.1:8000/predict"

# --- КООРДИНАТЫ ---
START_UE_COORDS = [0.0, 0.0, 100.0]
FINISH_UE_COORDS = [-100.0, 0.0, 100.0]
# FINISH_UE_COORDS = [-5015.0, -5.0, 50.0]

# --- НАСТРОЙКИ ПОЛЕТА ---
# Высота взлета (метры)
TAKEOFF_ALTITUDE = 10.0
# На каком расстоянии от цели выключать RL (метры)
RL_ARRIVAL_DISTANCE = 15.0
CAMERA_FOV = 90.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# --- НАСТРОЙКИ ПОСАДКИ ---
# Максимальный допустимый угол наклона поверхности
MAX_SLOPE_DEGREES = 15.0
# Скорость снижения при посадке (м/с)
LANDING_SPEED_Z = 0.5
# Минимальный радиус круга в пикселях для безопасной посадки
MIN_SAFE_ZONE_RADIUS = 20

# --- ПИД РЕГУЛЯТОР ---
PID_Kp = 0.005 
PID_Ki = 0.0001
PID_Kd = 0.005