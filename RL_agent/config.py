# ==========================================
# Конфигурация для обучения RL-агента
# ==========================================

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- НАСТРОЙКА ПЕРЕЗАПУСКА ---
# Имя текущего рана (если None, сгенерируется по времени)
# Если указать существующую папку (например "PPO_run_12_10_2023_14_00"), скрипт продолжит обучение
EXPERIMENT_NAME = None

LOAD_FROM_RUN_NAME = "PPO_drone_29_04_2026_08_17_19"
DEBUG = False

# --- ПУТИ ---
RAW_ROUTES_PATH = os.path.join(BASE_DIR, "raw_routes_config.json")
ROUTES_CONFIG_PATH = os.path.join(BASE_DIR, "routes_config.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TENSORBOARD_LOG = os.path.join(BASE_DIR, "ppo_drone_tensorboard")

# --- UNREAL ENGINE SETTINGS ---
UE_EXECUTABLE = r"D:\ProgramFiles\UE_5.4\Engine\Binaries\Win64\UnrealEditor.exe"
UE_PROJECT_PATH = r"D:\UnrealProjects\Diploma\Diploma.uproject"
UE_PROCESS_NAME = "UnrealEditor.exe"

# Максимальное время в секундах на 1 шаг.
# Если больше - убиваем процесс (защита от зависаний).
WATCHDOG_TIMEOUT = 120
TIME2WAIT = 20

# --- СИМУЛЯЦИЯ И ДРОН ---
MAX_SPEED = 5.0
MOVE_TIME = 0.1
ARRIVAL_DIST = 2.0
DISTANCE_CLIP_THR = 20.0
CLOSE2TARGET_REWARD_COEFF = 10.0

# Логика динамического таймаута
COEFF_ROUTE_STEPS = 2.5
BASE_ROUTE_STEPS = 50

# --- АРХИТЕКТУРА НЕЙРОСЕТИ ---
DEPTH_MAP_OUT_SIZE = 256
VECTOR_OUT_SIZE = 64
NN_PI_ARCH = [128, 128] # Слои для Actor
NN_VF_ARCH = [128, 128] # Слои для Critic

# --- CURRICULUM LEARNING ---
# Как часто (в шагах) проверяем возможность смены уровня
CL_CHECK_FREQ = 1024
# CL_CHECK_FREQ = 50
CL_SUCCESS_THRESHOLD = 0.8
CL_WINDOW_SIZE_PER_ROUTE = 20
# CL_WINDOW_SIZE_PER_ROUTE = 5
CL_MAX_STEPS_PER_LEVEL = 1_048_576 * 3
# CL_MAX_STEPS_PER_LEVEL = 50

# --- ГИПЕРПАРАМЕТРЫ PPO ---
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 40_000_000
ENT_COEF = 0.03