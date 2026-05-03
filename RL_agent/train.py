import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import os
import numpy as np
from collections import deque
import time
import threading
import json
import shutil
import sys
import types
from stable_baselines3.common.logger import configure

from env import ColosseumDroneEnv
import config

# Время последнего успешного шага
LAST_STEP_TIME = time.time()

def watchdog_thread():
	'''
	Функция параллельного потока для отслеживания зависаний Unreal Engine.
	'''
	global LAST_STEP_TIME
	while True:
		# Производим проверку каждые 10 секунд
		time.sleep(10)
		# Если с момента последнего шага прошло больше времени, чем разрешено в конфиге:
		if time.time() - LAST_STEP_TIME > config.WATCHDOG_TIMEOUT:
			print(f"[WATCHDOG] FROZEN! Убиваем Unreal и Python...")
			# Убиваем Unreal
			os.system(f"taskkill /F /IM {config.UE_PROCESS_NAME} 2>nul")
			# Жестко завершаем сам скрипт
			os._exit(1)


class DroneMultimodalExtractor(BaseFeaturesExtractor):
	def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = config.DEPTH_MAP_OUT_SIZE + config.VECTOR_OUT_SIZE):
		super().__init__(observation_space, features_dim)	

		# 1. Голова depth map	
		n_input_channels_depthmap = observation_space.spaces["depth"].shape[0]
	
		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels_depthmap, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
		)
		# размер выхода cnn
		with torch.no_grad():
			cnn_out_dim = self.cnn(torch.as_tensor(observation_space.spaces["depth"].sample()[None]).float()).shape[1]

		# 2. Голова vector
		n_input_channels_vector = observation_space.spaces["vector"].shape[0]
		self.vector_mlp = nn.Sequential(
			nn.Linear(n_input_channels_vector, config.VECTOR_OUT_SIZE),
			nn.ReLU(),
			nn.Linear(config.VECTOR_OUT_SIZE, config.VECTOR_OUT_SIZE),
			nn.ReLU()
		)

		# 3. Финальный выход после слияния
		self.linear = nn.Sequential(
			nn.Linear(cnn_out_dim + config.VECTOR_OUT_SIZE, features_dim),
			nn.ReLU()
		)

	def forward(self, observations):
		# Прогоняем данные по веткам
		depth_features = self.cnn(observations["depth"])
		vector_features = self.vector_mlp(observations["vector"])

		# Конкатенируем
		concat_features = torch.cat([depth_features, vector_features], dim=1)

		return self.linear(concat_features)


class CurriculumCallback(BaseCallback):
	def __init__(self, run_dir, check_freq=config.CL_CHECK_FREQ, success_threshold=config.CL_SUCCESS_THRESHOLD,
				 window_size_per_route=config.CL_WINDOW_SIZE_PER_ROUTE, max_steps_per_level=config.CL_MAX_STEPS_PER_LEVEL,
				 max_steps_per_route_unlock=config.CL_MAX_STEPS_PER_ROUTE_UNLOCK, verbose=1):
		'''
		:check_freq: с какой частотой алгоритм будет проверять, нужно ли переходить на следующий уровень или нет;
		:success_threshold: процент успешных полетов для каждого маршута, после которого происходит
							переход на следующий уровень;
		:window_size_per_route: количество последних попыток, от которых будет рассчитывать процент успешных полетов;
		:max_steps_per_level: порог ко количеству шагов для каждого уровня.
		'''
		super(CurriculumCallback, self).__init__(verbose)
		self.run_dir = run_dir
		self.check_freq = check_freq
		self.success_threshold = success_threshold
		self.window_size_per_route = window_size_per_route
		self.max_steps_per_level = max_steps_per_level
		self.max_steps_per_route_unlock = max_steps_per_route_unlock
		
		# Храним информацию для каждого маршрута об удачных попытках.
		self.route_buffers = {}

		self.steps_in_current_level = 0
		self.steps_since_last_unlock = 0

	def save_state(self, save_buffer=False):
		env = self.training_env.envs[0].unwrapped
		
		# Если запланирован переход на новый уровень, сохраняем его.
		# Иначе сохраняем текущий.
		actual_level = env.pending_level_change if env.pending_level_change is not None else env.current_level
		
		state_data = {
			"current_level": actual_level,
			"current_route_idx": env.current_route_idx,
			"unlocked_routes_count": getattr(env, "unlocked_routes_count", 1),
			"steps_in_current_level": self.steps_in_current_level,
			"steps_since_last_unlock": self.steps_since_last_unlock,
			"total_timesteps": self.num_timesteps,
			"route_buffers": {k: list(v) for k, v in self.route_buffers.items()} 
		}

		with open(os.path.join(self.run_dir, "state.json"), "w") as f:
			json.dump(state_data, f, indent=4)

		# На всякий случай сохраняем саму модель
		self.model.save(os.path.join(self.run_dir, "latest_model.zip"))

		if save_buffer and config.ALGORITHM == "SAC":
			print("\n[INFO] Сохранение Replay Buffer (несколько ГБ), подождите...")
			self.model.save_replay_buffer(os.path.join(self.run_dir, "replay_buffer.pkl"))

	def load_state(self, state_data):
		self.steps_in_current_level = state_data.get("steps_in_current_level", 0)
		self.steps_since_last_unlock = state_data.get("steps_since_last_unlock", 0)
		self.total_timesteps_passed = state_data.get("total_timesteps", 0)

		buffers = state_data.get("route_buffers", {})
		for k, v in buffers.items():
			self.route_buffers[int(k)] = deque(v, maxlen=self.window_size_per_route)

		self.num_timesteps = self.total_timesteps_passed

	def _on_step(self) -> bool:
		global LAST_STEP_TIME
		LAST_STEP_TIME = time.time()

		self.steps_in_current_level += 1
		self.steps_since_last_unlock += 1

		# Проверяем, закончился ли эпизод.
		# P.S.: у нас одна среда, поэтому индекс 0.
		if self.locals.get("dones")[0]:
			# данные с последнего шага
			info = self.locals.get("infos")[0]

			route_idx = info.get("route_idx", 0)
			is_success = info.get("is_success", False)

			if route_idx not in self.route_buffers:
				self.route_buffers[route_idx] = deque(maxlen=self.window_size_per_route)

			# Результат эпизода: 1 --> успех, 0 --> краш / таймаут
			self.route_buffers[route_idx].append(1.0 if is_success else 0.0)

		# Периодическое сохранение прогресса
		if self.num_timesteps % config.SAVE_FREQ_STEPS == 0:
			self.save_state()

		# == Проверка уровня и маршрутов ==
		if self.n_calls % self.check_freq == 0:
			env = self.training_env.envs[0].unwrapped
			current_lvl = env.current_level
			max_lvl = env.max_level

			total_routes_in_level = len(env.routes_data[f"level_{current_lvl}"])

			force_upgrade_level = False
			all_routes_passed = False
			force_unlock_route = False
			unlocked = getattr(env, "unlocked_routes_count", 1)

			if self.steps_in_current_level >= self.max_steps_per_level:
				force_upgrade_level = True
			else:
				# Считаем успешно пройденные маршруты
				passed_count = 0
				if len(self.route_buffers) >= unlocked:
					for r_idx in range(unlocked):
						buf = self.route_buffers.get(r_idx,[])
						if len(buf) == self.window_size_per_route and np.mean(buf) >= self.success_threshold:
							passed_count += 1
							
				# Открытие маршрута по винрейту
				if passed_count == unlocked:
					if unlocked < total_routes_in_level:
						force_unlock_route = "УСПЕХ (Отличный winrate)"
					else:
						all_routes_passed = True

				# Принудительное открытие по таймауту
				elif self.steps_since_last_unlock >= self.max_steps_per_route_unlock:
					if unlocked < total_routes_in_level:
						force_unlock_route = "ТАЙМАУТ (Слишком долго на маршруте)"

				# Вывод статистики в консоль
				if self.verbose > 0:
					stats_str =[]
					for r_idx in range(total_routes_in_level):
						buf = self.route_buffers.get(r_idx,[])
						rate = np.mean(buf) * 100 if len(buf) > 0 else 0.0
						stats_str.append(f"Маршрут {r_idx}: {rate:.0f}% ({len(buf)}/{self.window_size_per_route})")

					print(f"[{self.n_calls}] Level {current_lvl} | " + " | ".join(stats_str))
					print(f"Шагов до разблокировки: {self.steps_since_last_unlock}/{self.max_steps_per_route_unlock}")
					print(f"Шагов на уровне: {self.steps_in_current_level}/{self.max_steps_per_level}")

				if force_unlock_route:
					env.unlocked_routes_count += 1
					print(f"\n>>> ОТКРЫТ МАРШРУТ #{env.unlocked_routes_count - 1} | Причина: {force_unlock_route} <<<")
					self.route_buffers.clear()
					self.steps_since_last_unlock = 0
					self.save_state(save_buffer=False)

				# Переход на следующий уровень
				if all_routes_passed or force_upgrade_level:
					reason = "ТАЙМАУТ УРОВНЯ" if force_upgrade_level else "УСПЕХ (Все маршруты уровня освоены)"
				
					new_lvl = (current_lvl + 1) % (max_lvl + 1)

					print(f"\n=============================================")
					print(f"!!! {reason} !!!")
					print(f"Переход на уровень: level_{new_lvl}")
					print(f"=============================================\n")

					env.set_level(new_lvl)
					self.steps_in_current_level = 0
					self.steps_since_last_unlock = 0
					self.route_buffers.clear()

					self.save_state(save_buffer=True)
					sys.exit(42)
		
		return True


def dump_config(run_dir):
	'''
	Сохраняет текущие параметры config.py в папку модели.
	'''
	conf_dict = {}
	for k in dir(config):
		val = getattr(config, k)
		# Фильтрация по тому, что сохрянаем в конфиг:
		# 1. Не системное поле (начинается с __)
		# 2. Не функция/метод
		# 3. Не импортированный модуль
		if not k.startswith("__") and not callable(val) and not isinstance(val, types.ModuleType):
			conf_dict[k] = val

	with open(os.path.join(run_dir, "run_config.json"), "w", encoding='utf-8') as f:
		json.dump(conf_dict, f, indent=4, ensure_ascii=False)

def main():
	# Запуск Watchdog
	wd_thread = threading.Thread(target=watchdog_thread, daemon=True)
	wd_thread.start()

	run_name = os.environ.get("RUN_NAME")
	# для возможности запуска без менеджера:
	if not run_name:
		if config.EXPERIMENT_NAME:
			run_name = config.EXPERIMENT_NAME
		else:
			run_name = f"{config.ALGORITHM}_drone_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

	run_dir = os.path.join(config.MODELS_DIR, run_name)
	os.makedirs(run_dir, exist_ok=True)
	dump_config(run_dir) # сохраняем параметры

	# Если есть данные о маршрутах, то копируем их в архив папки
	if os.path.exists(config.ROUTES_CONFIG_PATH):
		shutil.copy(config.ROUTES_CONFIG_PATH, os.path.join(run_dir, "routes_data.json"))

	print("=== Инициализация среды ===")
	env = ColosseumDroneEnv()

	# Восстановление состояния после перезапуска
	# Путь к текущему рану (куда сохраняем)
	current_state_path = os.path.join(run_dir, "state.json")
	current_model_path = os.path.join(run_dir, "latest_model.zip")
	# Путь к "источнику" (откуда можем взять веса для старта)
	source_run_name = getattr(config, "LOAD_FROM_RUN_NAME", None)
	source_dir = os.path.join(config.MODELS_DIR, source_run_name) if source_run_name else None

	start_timesteps = 0
	loaded_state_data = None
	load_path = None

	# 1. Сначала проверяем, есть ли прогресс в текущей папке (продолжение прерванного обучения)
	if os.path.exists(current_state_path) and os.path.exists(current_model_path):
		print(f"=== Восстановление из текущей папки: {run_dir} ===")
		with open(current_state_path, "r") as f:
			loaded_state_data = json.load(f)
		load_path = current_model_path
	
	# 2. Если в текущей пусто, проверяем старую модель
	elif source_dir and os.path.exists(os.path.join(source_dir, "latest_model.zip")):
		print(f"=== Старт нового эксперимента на базе старого: {source_run_name} ===")
		source_state_path = os.path.join(source_dir, "state.json")
		if os.path.exists(source_state_path):
			with open(source_state_path, "r") as f:
				loaded_state_data = json.load(f)
		load_path = os.path.join(source_dir, "latest_model.zip")
	else:
		print("=== Старт обучения с чистого листа (веса инициализированы случайно) ===")
		load_path = None

	# Применяем загруженный стейт (если нашли хоть где-то)
	if loaded_state_data:
		env.load_state(
			loaded_state_data.get("current_level", 0),
			loaded_state_data.get("current_route_idx", 0),
			loaded_state_data.get("unlocked_routes_count", 1)
		)
		start_timesteps = loaded_state_data.get("total_timesteps", 0)

	env = Monitor(env)
	env = DummyVecEnv([lambda: env])

	if config.ALGORITHM == "PPO":
		net_arch = dict(pi=config.NN_PI_ARCH, vf=config.NN_VF_ARCH)
	elif config.ALGORITHM == "SAC":
		net_arch = dict(pi=config.NN_PI_ARCH, qf=config.NN_VF_ARCH)

	# Настройки архитектуры нейросети
	policy_kwargs = dict(
		features_extractor_class=DroneMultimodalExtractor,
		features_extractor_kwargs=dict(features_dim=config.DEPTH_MAP_OUT_SIZE + config.VECTOR_OUT_SIZE),
		# Слои для Actor (pi) и Critic (vf) после извлечения фичей.
		# 2 полносвязных слоя размером 128 для каждого.
		net_arch=net_arch
	)

	curriculum_callback = CurriculumCallback(run_dir=run_dir)
	AlgoClass = PPO if config.ALGORITHM == "PPO" else SAC

	if load_path:
		model = AlgoClass.load(load_path, env=env, tensorboard_log=config.TENSORBOARD_LOG)
		
		# Загрузка буфера для SAC
		buffer_path = os.path.join(os.path.dirname(load_path), "replay_buffer.pkl")
		if config.ALGORITHM == "SAC" and os.path.exists(buffer_path):
			print("=== Загрузка Replay Buffer (может занять время) ===")
			model.load_replay_buffer(buffer_path)

		if loaded_state_data:
			curriculum_callback.load_state(loaded_state_data)
	else:
		if config.ALGORITHM == "PPO":
			model = PPO(
				"MultiInputPolicy", 
				env,
				policy_kwargs=policy_kwargs,
				verbose=1,

				learning_rate=config.PPO_LEARNING_RATE,
				# сколько шагов собрать перед обновлением весов
				n_steps=config.PPO_N_STEPS,
				batch_size=config.PPO_BATCH_SIZE,
				n_epochs=config.PPO_N_EPOCHS,
				# параметр дисконтирования (насколько важны будущие награды)
				gamma=config.PPO_GAMMA,

				max_grad_norm=0.5,
				ent_coef=config.PPO_ENT_COEF,
			)
		elif config.ALGORITHM == "SAC":
			model = SAC(
				"MultiInputPolicy", 
				env,
				policy_kwargs=policy_kwargs,
				verbose=1,

				learning_rate=config.SAC_LEARNING_RATE,
				buffer_size=config.SAC_BUFFER_SIZE,
				learning_starts=config.SAC_LEARNING_STARTS,

				batch_size=config.SAC_BATCH_SIZE,
				train_freq=config.SAC_TRAIN_FREQ,
				gradient_steps=config.SAC_GRADIENT_STEPS,

				gamma=config.SAC_GAMMA,

				tau=config.SAC_TAU,
				ent_coef=config.SAC_ENT_COEF,
			)

	tb_log_dir = os.path.join(config.TENSORBOARD_LOG, run_name)
	new_logger = configure(tb_log_dir, ["stdout", "tensorboard"])
	model.set_logger(new_logger)

	print("=== Старт обучения ===")
	remaining_timesteps = config.TOTAL_TIMESTEPS - start_timesteps

	try:
		if remaining_timesteps > 0:
			model.learn(total_timesteps=remaining_timesteps, callback=curriculum_callback, progress_bar=True, reset_num_timesteps=False)
		else:
			print("Обучение уже завершено (достигнут TOTAL_TIMESTEPS).")
	except KeyboardInterrupt:
		print("Обучение прервано пользователем. Сохраняем модель...")
	except Exception as e:
		print(f"Обучение было прервано. Сохраняем модель...\n{e}")
		sys.exit(1)
	finally:
		curriculum_callback.save_state(save_buffer=True)
		env.close()


if __name__ == "__main__":
	main()