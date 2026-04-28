import os
import subprocess
import time
import sys
import psutil
import argparse
import json
from datetime import datetime
import itertools



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from settings_manager import ensure_settings
import config

def kill_unreal():
    print("[MANAGER] Убиваем процесс Unreal Engine (если запущен)...")
    os.system(f"taskkill /F /IM {config.UE_PROCESS_NAME} 2>nul")
    time.sleep(2)

def is_unreal_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == config.UE_PROCESS_NAME:
            return True
    return False

def start_unreal():
    if is_unreal_running():
        return
    print(f"[MANAGER] Запуск Unreal Engine на карте {config.MAP_NAME}...")
    
    cmd = f'start "" "{config.UE_EXECUTABLE}" "{config.UE_PROJECT_PATH}" {config.MAP_NAME} -game -windowed -ResX=800 -ResY=600 -NoSound'
    os.system(cmd)

    print(f"[MANAGER] Ожидание {config.TIME2WAIT} секунд для инициализации UE...")
    time.sleep(config.TIME2WAIT)

def main():
	parser = argparse.ArgumentParser(description="Автоматический менеджер (Сбор/Обучение/Посадка)")
	parser.add_argument('--mode', type=str, choices=['collect', 'train', 'search'], required=True)
	parser.add_argument('--resume_run', type=str, default=None, help="Имя папки эксперимента для продолжения (например: UNet_resnet34_12_05_2024...)")
	args = parser.parse_args()

	print(f"=== АВТОМАТИЧЕСКИЙ МЕНЕДЖЕР (РЕЖИМ: {args.mode.upper()}) ===")

	if args.mode == 'collect':
		ensure_settings(args.mode)

	# Генерация имени эксперимента для обучения.
	env_vars = os.environ.copy()
	if args.mode == 'train':
		import config_train

		if args.resume_run:
			run_name = args.resume_run
			print(f"[MANAGER] ПРОДОЛЖЕНИЕ эксперимента: {run_name}")
		else:
			timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
			run_name = f"{config_train.SEG_MODEL_NAME}_{config_train.SEG_BACKBONE}_{timestamp}"
			print(f"[MANAGER] Текущий эксперимент: {run_name}")
		env_vars["RUN_NAME"] = run_name

	elif args.mode == 'search':
		print("\n=== РЕЖИМ GRID SEARCH ===")

		# Логика возобновления или создания новой сессии
		if args.resume_run:
			session_name = args.resume_run
			session_dir = os.path.join(config.MODELS_DIR, session_name)
			print(f"[MANAGER] Продолжение Search-сессии: {session_name}")
			
			config_path = os.path.join(session_dir, "search_config.json")
			if os.path.exists(config_path):
				with open(config_path, "r", encoding="utf-8") as f:
					grids = json.load(f)
			else:
				print(f"[ОШИБКА] Не найден файл search_config.json в {session_dir}!")
				return
		else:
			import config_search
			
			timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
			session_name = f"Search_Session_{timestamp}"
			session_dir = os.path.join(config.MODELS_DIR, session_name)
			os.makedirs(session_dir, exist_ok=True)
			
			grids = config_search.SEARCH_GRIDS
			with open(os.path.join(session_dir, "search_config.json"), "w", encoding="utf-8") as f:
				json.dump(grids, f, indent=4)
			print(f"[MANAGER] Создана новая Search-сессия: {session_name}")

		experiments = []
		for grid in grids:
			keys, values = zip(*grid.items())
			for v in itertools.product(*values):
				experiments.append(dict(zip(keys, v)))
				
		print(f"[*] Всего экспериментов в очереди: {len(experiments)}")
		
		# Запуск очереди экспериментов
		for idx, exp in enumerate(experiments, 1):
			kwargs_dict = json.loads(exp.get("EXTRA_KWARGS", "{}"))
			kwargs_str = "_" + "_".join([f"{k}-{v}" for k, v in kwargs_dict.items()]) if kwargs_dict else ""
			
			exp_name = f"{exp['SEG_MODEL_NAME']}_{exp['SEG_BACKBONE']}_BS{exp['BATCH_SIZE']}_{exp['OPTIMIZER']}_LR{exp['LEARNING_RATE']}{kwargs_str}"
			
			run_name = f"{session_name}/{exp_name}"
			
			print(f"\n{'='*70}")
			print(f"[SEARCH] ЗАПУСК ЭКСПЕРИМЕНТА {idx}/{len(experiments)}")
			print(f"Модель: {exp_name}")
			print(f"{'='*70}")
			
			env_vars = os.environ.copy()
			env_vars["RUN_NAME"] = run_name
			
			# Загружаем в среду все параметры текущего эксперимента
			for k, v in exp.items():
				env_vars[k] = str(v)
				
			process = subprocess.Popen(
				[sys.executable, "main.py", "--mode", "train"],
				cwd=BASE_DIR,
				env=env_vars
			)
			process.wait()
			
			if process.returncode != 0:
				print(f"[MANAGER] Эксперимент '{exp_name}' упал с ошибкой! Пропускаем и идем дальше...")
				time.sleep(3)
				
		print("\n[MANAGER] ВСЕ ЭКСПЕРИМЕНТЫ ИЗ СЕТКИ ЗАВЕРШЕНЫ!")
		return

	while True:
		if args.mode in ['collect']:
			# Убиваем старый процесс и запускаем новый, чтобы UE точно подхватил settings.json
			kill_unreal()
			start_unreal()

		print(f"[MANAGER] Запуск рабочего скрипта main.py --mode {args.mode}...")
		
		process = subprocess.Popen(
			[sys.executable, "main.py", "--mode", args.mode],
			cwd=BASE_DIR,
			env=env_vars
		)
		
		process.wait()

		if process.returncode == 0:
			print(f"[MANAGER] Процесс '{args.mode}' успешно завершен!")
			if args.mode in ['collect']:
				kill_unreal()
			break
		else:
			print(f"[MANAGER] Скрипт завершился с ошибкой (код {process.returncode}). Перезапуск через 5 секунд...")
			time.sleep(5)

if __name__ == "__main__":
	main()