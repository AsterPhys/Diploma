# ==========================================
# Конфигурация для обучения RL-агента
# ==========================================

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- АЛГОРИТМ ---
ALGORITHM = "SAC" # "PPO" / "SAC"

# --- НАСТРОЙКА ПЕРЕЗАПУСКА ---
# Имя текущего рана (если None, сгенерируется по времени)
# Если указать существующую папку (например "PPO_run_12_10_2023_14_00"), скрипт продолжит обучение
EXPERIMENT_NAME = None

LOAD_FROM_RUN_NAME = "SAC_drone_05_05_2026_00_08_36"
DEBUG = False

# --- ПУТИ ---
RAW_ROUTES_PATH = os.path.join(BASE_DIR, "raw_routes_config.json")
ROUTES_CONFIG_PATH = os.path.join(BASE_DIR, "routes_config.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TENSORBOARD_LOG = os.path.join(BASE_DIR, "rl_tensorboard")

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
CL_SUCCESS_THRESHOLD = 0.6
CL_WINDOW_SIZE_PER_ROUTE = 20
CL_MAX_STEPS_PER_LEVEL = 400_384
CL_MAX_STEPS_PER_ROUTE_UNLOCK = 1024 * 50
SAVE_FREQ_STEPS = 1024
TOTAL_TIMESTEPS = 40_000_000

# --- ГИПЕРПАРАМЕТРЫ PPO ---
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_LEARNING_RATE = 3e-4
PPO_ENT_COEF = 0.03
PPO_N_STEPS = 1024

# --- ГИПЕРПАРАМЕТРЫ SAC ---
SAC_BUFFER_SIZE = 50_000
SAC_BATCH_SIZE = 128
SAC_LEARNING_RATE = 3e-4
SAC_LEARNING_STARTS = 1000 # Сколько шагов просто собирать случайный опыт перед обучением
SAC_TRAIN_FREQ = 1         # Обучаться каждый шаг
SAC_GRADIENT_STEPS = 1     # Сколько шагов градиентного спуска делать
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_ENT_COEF = "auto"

# --- НАСТРОЙКИ ШУМОВ ---
POS_NOISE_STD = 0.2        # Стандартное отклонение позиционирования (метры)
DEPTH_NOISE_STD = 0.1      # Погрешность камеры глубины (метры)
DROPOUT_PROB = 0.01        # Вероятность "битого" пикселя глубины (1%)