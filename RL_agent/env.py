import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import math
import pymap3d as pm
from scipy.spatial.transform import Rotation as R
import json
import random
from collections import deque

from config import (
    ROUTES_CONFIG_PATH, MAX_SPEED, MOVE_TIME, 
    ARRIVAL_DIST, CLOSE2TARGET_REWARD_COEFF, DISTANCE_CLIP_THR,
	DEBUG
)

class ColosseumDroneEnv(gym.Env):
	def __init__(self):
		super(ColosseumDroneEnv, self).__init__()

		print("Подключение к AirSim...")
		self.client = airsim.MultirotorClient()

		connected = False
		n_tries = 20
		for attempt in range(n_tries):
			try:
				self.client.confirmConnection()
				connected = True
				print("Успешное подключение к AirSim!")
				break
			except Exception as e:
				print(f"  Ожидание сервера AirSim (попытка {attempt+1}/{n_tries})...")
				time.sleep(3)
		
		if not connected:
			raise RuntimeError("КРИТИЧЕСКАЯ ОШИБКА: AirSim не ответил.")
		time.sleep(1)
		
		drone_state = self.client.getMultirotorState()
		OriginGeopoint = drone_state.gps_location
		self.lat_start = OriginGeopoint.latitude
		self.lon_start = OriginGeopoint.longitude
		self.alt_start = OriginGeopoint.altitude
		
		# Загружаем маршруты для обучения на разных уровнях
		with open(ROUTES_CONFIG_PATH, 'r') as f:
			self.routes_data = json.load(f)

		# Логика переключения уровней по уровню сложности:
		# по ходу обучения, когда робот становится лучше, он
		# переходит на следующую локацию.
		self.max_level = len(self.routes_data) - 1
		self.current_level = 0
		self.current_route_idx = 0  # индекс текущего маршрута внутри уровня
		self.pending_level_change = 0 # Флаг для загрузки новой карты
		self.unlocked_routes_count = 1

		# Инициализируем переменные, которые будут обновляться в reset()
		self.target_position = np.zeros(3)
		self.MAX_DISTANCE = 1.0
		self.current_route_max_steps = 1024
		self.last_episode_success = False

		# Настройки для склеивания кадров карт глубины (для учета временного лага)
		self.n_frames = 4
		self.frames_buffer = deque(maxlen=self.n_frames)

		# Настройки
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
		self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0.0, high=1.0, shape=(self.n_frames, 84, 84), dtype=np.float32),
            "vector": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })

	def load_state(self, level, route_idx, unlocked_routes_count=1):
		'''
		Метод для восстановления state после перезапуска.
		'''
		self.current_level = level
		self.current_route_idx = route_idx
		self.pending_level_change = level
		self.unlocked_routes_count = unlocked_routes_count

	def _get_coords2vector(self, drone_position, drone_orientation, target_position, eps=1e-5):
		'''
		1. Переводим координаты из World Frame в Body Frame.
		2. Рассчитываем вектор (проекции на все оси и модуль расстояния).
		Возвращает [X_body, Y_body, Z_body, distance]
		'''
		# вектор до цели в World Frame
		V_world = target_position - drone_position
		distance = np.linalg.norm(V_world)

		# если мы уже долетели до цели, может возникнуть ситуация
		# деления на 0
		if distance < eps:
			return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

		rotation = R.from_quat(drone_orientation)
		# получаем вектор до цели в Body Frame
		V_body = rotation.inv().apply(V_world)

		V_body_normalized = V_body / distance
		distance_normalized = np.clip(distance / self.MAX_DISTANCE, 0.0, 1.0)

		vector = np.concatenate((V_body_normalized, [distance_normalized]))
		
		return vector

	def _get_obs(self):
		# -- 1. Получаем карту глубины --
		response = self.client.simGetImages([
			airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
		])[0]
		# !!! не забыть настроить камеру, чтобы она возвращала изображение 84x84
		data_img = airsim.list_to_2d_float_array(response.image_data_float, \
												 response.width, response.height)
		data_depth_map = np.clip(data_img / DISTANCE_CLIP_THR, 0.0, 1.0)
		data_depth_map = np.expand_dims(data_depth_map, axis=0).astype(np.float32)

		# -- 2. Получаем положение дрона --
		# !!! не забыть настроить шумы
		state_estimated = self.client.getMultirotorState().kinematics_estimated
		position = state_estimated.position
		orientation = state_estimated.orientation
		
		drone_position = np.array([position.x_val, position.y_val, position.z_val])
		drone_orientation = [orientation.x_val, orientation.y_val, \
							 orientation.z_val, orientation.w_val]
		
		data_vector = self._get_coords2vector(drone_position, drone_orientation, self.target_position)

		return {
			"depth": data_depth_map,
			"vector": data_vector
		}

	def _generate_new_route(self):
		# 1. Получаем текущий уровень и маршрут
		level_key = f"level_{self.current_level}"
		route = self.routes_data[level_key][self.current_route_idx]
		self.current_route_max_steps = route.get("max_steps", 1000)

		# 2. Получаем точку старта в NED координатах и GPS-координаты финиша,
		# которые переводим в NED
		start_coords = route["start_local"]
		start_position = np.array([start_coords[0], start_coords[1], start_coords[2]])

		random_yaw = random.uniform(math.radians(-15), math.radians(15))
		start_vector = airsim.Vector3r(float(start_position[0]), float(start_position[1]), float(start_position[2]))
		start_orientation = airsim.to_quaternion(0, 0, random_yaw)
		start_pose = airsim.Pose(start_vector, start_orientation)

		target_gps = route["target_gps"]
		target_ned = pm.geodetic2ned(
			target_gps[0], target_gps[1], target_gps[2],
			self.lat_start, self.lon_start, self.alt_start
		)
		target_position = np.array(target_ned)
		
		# 3. Рассчитываем дистанцию
		self.MAX_DISTANCE = np.linalg.norm(target_position - start_position)
		if self.MAX_DISTANCE < 1.0:
			self.MAX_DISTANCE = 1.0

		return start_position, start_pose, target_position

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
        
		if options is not None:
			if "level" in options:
				self.current_level = options["level"]
			if "route" in options:
				self.current_route_idx = options["route"]
		else:
			# Смена уровня по необходимости
			if self.pending_level_change is not None:
				self.current_level = self.pending_level_change
				self.pending_level_change = None

			# Выбираем случайный маршрут из разблокированных
			self.current_route_idx = random.randint(0, self.unlocked_routes_count - 1)

        # Сбрасываем симулятор и дрона
		self.client.simPause(False)
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)

		self.last_episode_success = False

		# Устанавливаем маршрут
		self.start_position, self.start_pose, self.target_position = self._generate_new_route()
		
		self.client.simSetVehiclePose(self.start_pose, True)

		self.client.moveByVelocityAsync(0.0, 0.0, 0.0, duration=0.2)
		self.client.simContinueForTime(0.2)
		_ = self.client.simGetCollisionInfo()

		self.current_step = 0
		self.prev_distance = self.MAX_DISTANCE

		self.frames_buffer.clear()

        # Получаем первое наблюдение
		observation = self._get_obs()

		# Забиваем буфер 4 копиями первого кадра
		for _ in range(self.n_frames):
			self.frames_buffer.append(observation["depth"])
		observation["depth"] = np.concatenate(self.frames_buffer, axis=0)

		info = {
			"start": self.start_position,
			"finish": self.target_position,
			"level": self.current_level,
			"route_idx": self.current_route_idx
		}

		return observation, info

	def step(self, action):
		'''
		Нейросеть возвращает действие, функция возвращает результат действия.
		'''
		self.current_step += 1

		# Нейросеть возвращает нам скорости в Body Frame относительно дрона,
		# пересчитываем их в World Frame, чтобы дать команду дрону,
		# куда лететь.
		vx, vy, vz = action * MAX_SPEED
		V_body = np.array([vx, vy, vz])
		orientation = self.client.getMultirotorState().kinematics_estimated.orientation
		drone_orientation = [orientation.x_val, orientation.y_val, 
							 orientation.z_val, orientation.w_val]

		rotation = R.from_quat(drone_orientation)
		V_world = rotation.apply(V_body)
		vx, vy, vz = float(V_world[0]), float(V_world[1]), float(V_world[2])

		# !!! настроить ClockSpeed для ускорения симуляции и NoDisplay
		self.client.moveByVelocityAsync(vx, vy, vz, duration=MOVE_TIME)
		self.client.simContinueForTime(MOVE_TIME)

		observation = self._get_obs()

		self.frames_buffer.append(observation["depth"])
		observation["depth"] = np.concatenate(self.frames_buffer, axis=0)

		# ---

		terminated = False
		truncated = False
		route_idx_temp = self.current_route_idx
		
		current_distance = observation["vector"][3] * self.MAX_DISTANCE
		progress_reward = self.prev_distance - current_distance
		self.prev_distance = current_distance

		reward = progress_reward * CLOSE2TARGET_REWARD_COEFF
		reward -= 0.05

		# 1. Условие успеха (мы долетели)
		if current_distance < ARRIVAL_DIST:
			terminated = True
			reward += 100

			self.last_episode_success = True
			# # Если агент успешно долетел, в следующем эпизоде даем ему следующий маршрут
			# level_key = f"level_{self.current_level}"
			# routes_in_current_level = len(self.routes_data[level_key])
			# self.current_route_idx = (self.current_route_idx + 1) % routes_in_current_level
			print(f"[ENV] Успех! Маршрут #{self.current_route_idx} пройден!")
			
		# 2. Условие провала (врезались)
		collision_info = self.client.simGetCollisionInfo()
		if collision_info.has_collided:
			if DEBUG == True:
				print(f"[DEBUG] Врезался в: {collision_info.object_name} на шаге {self.current_step}")
			terminated = True
			reward = -100

		# 3. Условие завершения по времени
		if self.current_step >= self.current_route_max_steps:
			truncated = True

		info = {
			"is_success": getattr(self, 'last_episode_success', False),
			"route_idx": route_idx_temp
		}

		return observation, reward, terminated, truncated, info

	def set_level(self, new_level):
		'''
		Вызывается извне, когда пора сменить локацию.
		'''
		if new_level <= self.max_level:
			print(f"\n=============================================")
			print(f">>> АГЕНТ ПОВЫШЕН! ПЕРЕХОД НА ЛОКАЦИЮ: level_{new_level} <<<")
			print(f"=============================================\n")
			self.pending_level_change = new_level
			self.current_route_idx = 0 # сбрасываем маршрут
			self.unlocked_routes_count = 1 # сбрасываем пул маршрутов

	def close(self):
		try:
			self.client.simPause(False)
			self.client.armDisarm(False)
			self.client.enableApiControl(False)
		except:
			pass