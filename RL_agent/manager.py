import os
import subprocess
import time
import sys
import psutil
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import config
from utils import generate_config
from settings_manager import ensure_settings

def kill_unreal():
	print("[MANAGER] Killing Unreal Engine...")
	os.system(f"taskkill /F /IM {config.UE_PROCESS_NAME} 2>nul")
	time.sleep(2)

def is_unreal_running():
	for proc in psutil.process_iter(['name']):
		if proc.info['name'] == config.UE_PROCESS_NAME:
			return True
	return False

def get_current_map_name(run_name):
	'''
	Определяет имя текущей карты исходя из сохраненного прогресса.
	'''
	current_level = 0
	state_path = os.path.join(config.MODELS_DIR, run_name, "state.json")
	
	if not os.path.exists(state_path) and getattr(config, "LOAD_FROM_RUN_NAME", None):
		source_state_path = os.path.join(config.MODELS_DIR, config.LOAD_FROM_RUN_NAME, "state.json")
		if os.path.exists(source_state_path):
			state_path = source_state_path
	
	if os.path.exists(state_path):
		with open(state_path, "r") as f:
			state_data = json.load(f)
			current_level = state_data.get("current_level", 0)

	with open(config.ROUTES_CONFIG_PATH, "r") as f:
		routes_data = json.load(f)

	level_key = f"level_{current_level}"
	if routes_data.get(level_key):
		return routes_data[level_key][0].get("map_name", "BaseLevel")
	
	return "BaseLevel"

def start_unreal(map_name):
	if is_unreal_running():
		return
	print(f"[MANAGER] Starting Unreal Engine on map {map_name}...")
	
	cmd = f'start "" "{config.UE_EXECUTABLE}" "{config.UE_PROJECT_PATH}" {map_name} -game -windowed -ResX=800 -ResY=600 -NoSound'
	os.system(cmd)

	print(f"Waiting {config.TIME2WAIT} seconds for Unreal initialization...")
	time.sleep(config.TIME2WAIT)


def main():
	print("=== АВТОМАТИЧЕСКИЙ МЕНЕДЖЕР ОБУЧЕНИЯ ===")
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	ensure_settings("rl_train")
	if not os.path.exists(config.ROUTES_CONFIG_PATH):
		generate_config(start_unreal, kill_unreal)

	if config.EXPERIMENT_NAME:
		run_name = config.EXPERIMENT_NAME
	else:
		run_name = f"{config.ALGORITHM}_drone_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
	print(f"[MANAGER] Текущий эксперимент: {run_name}")

	env_vars = os.environ.copy()
	env_vars["RUN_NAME"] = run_name

	while True:
		kill_unreal()

		target_map = get_current_map_name(run_name)
		start_unreal(target_map)

		print("[MANAGER] Запуск скрипта train.py...")
		process = subprocess.Popen(
			[sys.executable, "train.py"],
			cwd=BASE_DIR,
			env=env_vars
		)
		
		# Ждем завершения скрипта
		process.wait()

		# Если код возврата 0 - скрипт сам закончил обучение чисто.
		if process.returncode == 0:
			print("[MANAGER] Обучение успешно завершено!")
			kill_unreal()
			break
		elif process.returncode == 42:
			print("[MANAGER] АГЕНТ ПОВЫСИЛ УРОВЕНЬ! Перезапуск среды с новой картой...")
			time.sleep(2)
		else:
			print(f"[MANAGER] Скрипт завершился с ошибкой (код {process.returncode}). Перезапуск через 5 секунд...")
			time.sleep(5)


if __name__ == "__main__":
	main()