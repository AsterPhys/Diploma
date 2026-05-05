import airsim
import numpy as np
import time
import pymap3d as pm
from scipy.spatial.transform import Rotation as R

class DroneController:
	def __init__(self):
		# Подключение к симулятору
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)

		# Переменные для ПИД-регулятора (хранят накопленную ошибку)
		self.err_sum_x = 0
		self.err_prev_x = 0
		self.err_sum_y = 0
		self.err_prev_y = 0
		
		# Флаг первого запуска ПИД-регулятора
		self.first_pid_run = True 
		# Время последнего шага для расчета высоты
		self.last_time = time.time()

	# --- НАВИГАЦИЯ И КООРДИНАТЫ ---

	def ue_to_ned(self, start_ue, target_ue):
		"""
		Конвертирует глобальные координаты Unreal Engine
		в локальные координаты NED.
		"""     
		dx = target_ue[0] - start_ue[0]
		dy = target_ue[1] - start_ue[1]
		dz = target_ue[2] - start_ue[2]
		
		return [dx / 100.0, dy / 100.0, -dz / 100.0]

	def get_target_gps(self, target_ned):
		"""
		Переводит локальные координаты цели (NED) в глобальные GPS-координаты.
		Использует текущую точку старта дрона (Origin) как базу.
		"""
		# Получаем GPS-координаты точки старта (Origin) из симулятора
		state = self.client.getMultirotorState()
		origin = state.gps_location
		
		# Переводим локальные метры в гпс-координаты
		target_lat, target_lon, target_alt = pm.ned2geodetic(
			target_ned[0], target_ned[1], target_ned[2],
			origin.latitude, origin.longitude, origin.altitude
		)
		
		return target_lat, target_lon, target_alt

	def get_distance_to_target(self, target_ned):
		"""Считает расстояние от текущей позиции до цели в NED."""
		state = self.client.getMultirotorState()
		pos = state.kinematics_estimated.position
		
		curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
		targ_pos = np.array(target_ned)
		
		return np.linalg.norm(curr_pos - targ_pos)

	# --- УПРАВЛЕНИЕ ---

	def arm_and_takeoff(self, altitude_meters):
		"""Взлет на заданную высоту."""
		print(f"[CONTROL] Взлет на {altitude_meters}м...")
		self.client.takeoffAsync().join()
		self.client.moveToZAsync(-altitude_meters, 5).join()

	def move_by_action(self, action, max_speed=5):
		"""
		Выполняет действие от RL-агента.
		action: [vx, vy, vz]
		"""
		# [[x,y,z]] -> [x,y,z]
		action = np.squeeze(action) 

		# Получаем скорости в локальной системе дрона (Body Frame)
		vx_body = float(action[0]) * max_speed
		vy_body = float(action[1]) * max_speed
		vz_body = float(action[2]) * max_speed
		v_body = np.array([vx_body, vy_body, vz_body])

		# Получаем текущую ориентацию дрона
		state = self.client.getMultirotorState()
		orientation = state.kinematics_estimated.orientation
		drone_quat = [orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val]

		#  Переводим вектор из Body Frame в World Frame
		rotation = R.from_quat(drone_quat)
		v_world = rotation.apply(v_body)

		vx_w = float(v_world[0])
		vy_w = float(v_world[1])
		vz_w = float(v_world[2])
		
		# Делаем duration большим, чтобы дрон летел более плавно.
		# Это нужно для того, чтобы дрон не тормозил между командами, 
		# следующая команда все равно перезапишет текущую на лету.
		self.client.moveByVelocityAsync(vx_w, vy_w, vz_w, duration=10.0)

	def stop_moving(self):
		"""Мгновенная остановка."""
		self.client.moveByVelocityAsync(0, 0, 0, 1).join()

	def get_altitude(self):
		"""Возвращает текущую высоту от точки старта (положительное число)."""
		z = self.client.getMultirotorState().kinematics_estimated.position.z_val
		return abs(z)

	# --- ПИД-РЕГУЛЯТОР ДЛЯ ПОСАДКИ ---

	def pid_compute_velocity(self, target_u, target_v, img_w, img_h, kp, ki, kd):
		"""
		Вычисляет необходимые скорости vx, vy для центрирования дрона над точкой.
		target_u, target_v: координаты цели в пикселях на изображении с нижней камеры.
		"""
		curr_time = time.time()
		dt = curr_time - self.last_time
		if dt <= 0 or dt > 0.5:
			dt = 0.1

		# Находим центр экрана
		center_u = img_w / 2.0
		center_v = img_h / 2.0

		# Ошибка (отклонение цели от центра)
		# Камера смотрит вниз: 
		# Ошибка по вертикали картинки (v) соответствует движению ВПЕРЕД (X)
		# Ошибка по горизонтали картинки (u) соответствует движению ВПРАВО (Y)
		err_x = center_v - target_v  # Если цель выше центра по v, летим вперед (+X)
		err_y = target_u - center_u  # Если цель правее центра по u, летим вправо (+Y)

		if self.first_pid_run:
			self.err_prev_x = err_x
			self.err_prev_y = err_y
			self.first_pid_run = False

		# ПИД для X (Вперед/Назад)
		self.err_sum_x += err_x * dt
		d_err_x = (err_x - self.err_prev_x) / dt
		vx = (kp * err_x) + (ki * self.err_sum_x) + (kd * d_err_x)
		self.err_prev_x = err_x

		# ПИД для Y (Вправо/Влево)
		self.err_sum_y += err_y * dt
		d_err_y = (err_y - self.err_prev_y) / dt
		vy = (kp * err_y) + (ki * self.err_sum_y) + (kd * d_err_y)
		self.err_prev_y = err_y

		self.last_time = curr_time

		# Ограничение скорости для стабильности
		vx = np.clip(vx, -1, 1)
		vy = np.clip(vy, -1, 1)

		return vx, vy

	def move_velocity_ned(self, vx, vy, vz):
		"""Отправка команды скорости в симулятор."""
		self.client.moveByVelocityAsync(vx, vy, vz, 
										duration=1.0, 
										drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
										yaw_mode=airsim.YawMode(True, 0))

	# --- СЕНСОРЫ ---

	def get_camera_image(self, camera_name):
		"""
		Запрашивает RGB кадр из AirSim.
		Возвращает numpy массив (BGR).
		"""
		responses = self.client.simGetImages([
			airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
		])
		response = responses[0]
		
		img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
		img_rgb = img1d.reshape(response.height, response.width, 3)
		
		return img_rgb

	def get_camera_depth(self, camera_name):
		"""
		Запрашивает карту метрической глубины из AirSim.
		Возвращает 2D numpy массив float32 (в метрах).
		"""
		responses = self.client.simGetImages([
			airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
		])
		response = responses[0]
		
		depth_img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
		return depth_img

	def disarm(self):
		"""Выключить моторы."""
		self.client.armDisarm(False)
		self.client.enableApiControl(False)