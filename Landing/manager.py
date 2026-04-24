import os
import subprocess
import time
import sys
import psutil
import argparse

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
    parser.add_argument('--mode', type=str, choices=['collect', 'train', 'land'], required=True)
    args = parser.parse_args()

    print(f"=== АВТОМАТИЧЕСКИЙ МЕНЕДЖЕР (РЕЖИМ: {args.mode.upper()}) ===")

    ensure_settings(args.mode)

    while True:
        # Убиваем старый процесс и запускаем новый, чтобы UE точно подхватил settings.json
        kill_unreal()
        start_unreal()

        print(f"[MANAGER] Запуск рабочего скрипта main.py --mode {args.mode}...")
        
        process = subprocess.Popen([sys.executable, "main.py", "--mode", args.mode],
            cwd=BASE_DIR
        )
        
        process.wait()

        if process.returncode == 0:
            print(f"[MANAGER] Процесс '{args.mode}' успешно завершен!")
            kill_unreal()
            break
        else:
            print(f"[MANAGER] Скрипт завершился с ошибкой (код {process.returncode}). Перезапуск через 5 секунд...")
            time.sleep(5)

if __name__ == "__main__":
    main()