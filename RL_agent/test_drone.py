import os
import argparse
import time
from stable_baselines3 import PPO
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from env import ColosseumDroneEnv
from train import DroneMultimodalExtractor
from settings_manager import ensure_settings

# python test_drone.py --model models/PPO_drone_23_04_2026_09_55_59/latest_model.zip --level 0 --route 2 --episodes 3
def main():
	parser = argparse.ArgumentParser(description="Тестирование обученного агента")
	parser.add_argument("--model", type=str, required=True, help="Путь к .zip файлу модели (например, models/run_1/latest_model.zip)")
	parser.add_argument("--level", type=int, default=0, help="Индекс уровня (0 - BaseLevel, 1 - CityLevel и т.д.)")
	parser.add_argument("--routes", type=int, nargs='+', default=[0], help="Список маршрутов через пробел (например: --routes 0 2 5)")
	parser.add_argument("--fps", type=int, default=60, help="Задержка (в кадрах в секунду) для удобного просмотра")
	
	args = parser.parse_args()

	settings_changed = ensure_settings("rl_test")
	if settings_changed:
		print("\n" + "!" * 60)
		print("ВНИМАНИЕ: Настройки AirSim были изменены на режим 'rl_test'.")
		print("Если Unreal Engine СЕЙЧАС ОТКРЫТ, закрой его и открой заново,")
		print("иначе настройки (ClockSpeed=1.0, графика) не применятся!")
		print("!" * 60 + "\n")
		time.sleep(3)

	if not os.path.exists(args.model):
		raise FileNotFoundError(f"Модель не найдена по пути: {args.model}")

	print(f"=== Загрузка модели из {args.model} ===")
	model = PPO.load(args.model)

	print("=== Инициализация среды ===")
	env = ColosseumDroneEnv()

	for ep, route_idx in enumerate(args.routes):
		print(f"\n--- Эпизод {ep+1}/{len(args.routes)} | Уровень: {args.level} | Маршрут: {route_idx} ---")
		
		obs, info = env.reset(options={
			"level": args.level, 
			"route": route_idx
		})

		done = False
		total_reward = 0.0
		steps = 0

		while not done:
			# deterministic=True заставляет агента выбирать лучшее действие без режима исследования (exploration)
			action, _states = model.predict(obs, deterministic=True)
			
			obs, reward, terminated, truncated, info = env.step(action)
			total_reward += reward
			steps += 1
			
			done = terminated or truncated

			time.sleep(1.0 / args.fps)
			
		if info.get("is_success"):
			print(f"РЕЗУЛЬТАТ: УСПЕХ! Дрон достиг цели.")
		else:
			print(f"РЕЗУЛЬТАТ: КРАШ / ТАЙМАУТ.")
			
		print(f"Награда: {total_reward:.2f} | Шагов выжито: {steps}")

	env.close()
	print("\nТестирование завершено.")

if __name__ == "__main__":
	main()