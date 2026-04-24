import airsim
import cv2
import numpy as np
import os
import random
import math
import time
import config

class DataCollector:
	def __init__(self):
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()

		print("[COLLECTOR] Настройка классов сегментации AirSim...")
		# 0. Сбрасываем все в класс 0 (черный цвет - unknown/background).
		# На случай, если случайно пропущен какой-то мелкий проп и у него нет префикса, 
		# он стал черным (опасным), а не слился с безопасной землей.
		# Ну и + фон (небо) черный.
		self.client.simSetSegmentationObjectID(".*", 0, is_name_regex=True)
		# 1. Static_Obstacle -> 1
		self.client.simSetSegmentationObjectID(".*Static_Obstacle.*", 1, is_name_regex=True)
		# 2. Dynamic_Obstacle -> 2
		self.client.simSetSegmentationObjectID(".*Dynamic_Obstacle.*", 2, is_name_regex=True)
		# 3. Hazard -> 3
		self.client.simSetSegmentationObjectID(".*Hazard.*", 3, is_name_regex=True)
		# 4. Vegetation -> 4
		self.client.simSetSegmentationObjectID(".*Vegetation.*", 4, is_name_regex=True)
		# 5. Safe_Ground -> 5
		self.client.simSetSegmentationObjectID(".*Safe_Ground.*", 5, is_name_regex=True)
		# фикс бага с декалями
		self.client.simSetSegmentationObjectID(".*Decal.*", 0, is_name_regex=True)
		print("[COLLECTOR] Классы сегментации успешно назначены!")
		
		os.makedirs(config.RGB_DIR, exist_ok=True)
		os.makedirs(config.DEPTH_DIR, exist_ok=True)
		os.makedirs(config.MASK_DIR, exist_ok=True)
		os.makedirs(config.DEPTH_VIS_DIR, exist_ok=True)
		os.makedirs(config.MASK_VIS_DIR, exist_ok=True) 

		# --- ПОДГОТОВКА ГЕОМЕТРИИ ДЛЯ СПАВНА ---
		# OpenCV принимает numpy-массив для генерации контура (многоугольника)
		self.polygon = np.array(config.SPAWN_POLYGON, dtype=np.float32)
		
		# Находим крайние точки (bounding box) нашего многоугольника
		# Это нужно, чтобы сначала случайно сгенерировать точку в квадрате, а затем проверять,
		# попала ли она в область
		self.min_x, self.min_y = np.min(self.polygon, axis=0)
		self.max_x, self.max_y = np.max(self.polygon, axis=0)

	def randomize_environment(self):
		"""Меняет освещение и параметры погоды в симуляторе"""
		
		# 1. СМЕНА ВРЕМЕНИ СУТОК
		# Генерируем случайный час
		hour = random.randint(config.TIME_OF_DAY_MIN, config.TIME_OF_DAY_MAX)
		time_str = f"2026-04-24 {hour:02d}:00:00"
		# is_enabled - контроль за временем суток
		# start_datetime - точная дата и время
		# is_start_datetime_dst - переход на летнее время (неважно для нас)
		# celestial_clock_speed - скорость течения времени на небе (1 - реальное время, 
		# 60 - минута за секунду, 0 - заморозка времени)
		# update_interval_secs - скорость пересчета солнца (неважно при заморозке)
		# move_sun - разрешает двигать солнце физически, чтобы тени
		# тоже менялись
		self.client.simSetTimeOfDay(True, time_str, False, 0.0, 1.0, True)

		# 2. СБРОС ПОГОДЫ
		self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
		self.client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness, 0.0)
		self.client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, 0.0)
		self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0.0)
		self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)

		# 3. ВЫБОР НОВОГО ПОГОДНОГО ПРЕСЕТА
		# Случайно выбираем один из 4 вариантов
		weather_type = random.choice(["clear", "rain", "snow", "fog"])

		if weather_type == "clear":
			pass
			
		elif weather_type == "rain":
			# Дождь и мокрый асфальт (создает блики)
			self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, random.uniform(0.1, 0.6))
			self.client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness, random.uniform(0.4, 0.9))
			
		elif weather_type == "snow":
			# Снег на земле (делает текстуры белыми)
			self.client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, random.uniform(0.2, 0.6))
			
		elif weather_type == "fog":
			# Легкий туман и пыль (немного мылит картинку)
			self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, random.uniform(0.1, 0.4))
			self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, random.uniform(0.05, 0.15))

	def get_random_pose(self):
		"""Генерирует случайную позицию СТРОГО внутри полигона карты"""
		
		while True:
			x = random.uniform(self.min_x, self.max_x)
			y = random.uniform(self.min_y, self.max_y)
			
			# Проверяем расстояние до ближайшей границы полигона.
			# Положительное число означает, что точка внутри.
			dist = cv2.pointPolygonTest(self.polygon, (x, y), measureDist=True)

			# Если мы внутри полигона И расстояние до стены больше безопасного отступа
			if dist >= config.SAFE_MARGIN:
				break

		z = random.uniform(config.Z_MIN, config.Z_MAX)

		yaw = random.uniform(0, 360)
		pitch = random.uniform(-config.ROLL_PITCH_NOISE, config.ROLL_PITCH_NOISE)
		roll = random.uniform(-config.ROLL_PITCH_NOISE, config.ROLL_PITCH_NOISE)

		orientation = airsim.to_quaternion(math.radians(pitch), math.radians(roll), math.radians(yaw))
		position = airsim.Vector3r(x, y, z)
		
		return airsim.Pose(position, orientation)

	def collect_data(self, num_samples):
		print(f"Начинаем сбор {num_samples} кадров...")
		
		self.client.simPause(True)
		
		try:
			# --- WARM-UP ---
			print("[COLLECTOR] Прогрев симулятора и инициализация камеры...")
			warmup_pose = self.get_random_pose()
			self.client.simSetVehiclePose(warmup_pose, ignore_collision=True)
			
			if config.ENABLE_ENV_RANDOMIZATION:
				self.randomize_environment()
			
			self.client.simContinueForFrames(30)
			self.client.simGetImages([
				airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Scene, False, False)
			])
			print("[COLLECTOR] Прогрев завершен. Начинаем запись.")

			for i in range(num_samples):
				pose = self.get_random_pose()
				self.client.simSetVehiclePose(pose, ignore_collision=True)
				
				# Прокручиваем симуляцию
				self.client.simContinueForFrames(1)

				# Запрашиваем 3 картинки
				responses = self.client.simGetImages([
					airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Scene, False, False),
					airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.DepthPlanar, True, False),
					airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Segmentation, False, False)
				])

				# --> Сохраняем RGB
				img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
				img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
				cv2.imwrite(os.path.join(config.RGB_DIR, f"{i:05d}.png"), img_rgb)

				# --> Сохраняем Depth
				depth_img = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)
				np.save(os.path.join(config.DEPTH_DIR, f"{i:05d}.npy"), depth_img)

				# --> Сохраняем визуализацию карты глубины
				# Нормализуем массив (переводим float метры в диапазон 0-255).
				depth_norm = cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
				depth_vis = np.uint8(depth_norm)
				# Применяем тепловую карту для красоты.
				# Близкие объекты будут синими, средние - зелеными/желтыми, далекие - красными.
				depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
				cv2.imwrite(os.path.join(config.DEPTH_VIS_DIR, f"{i:05d}.png"), depth_colormap)

				# --> Сохраняем Segmentation
				mask1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
				img_mask_rgb = mask1d.reshape(responses[2].height, responses[2].width, 3)
				cv2.imwrite(os.path.join(config.MASK_VIS_DIR, f"{i:05d}.png"), img_mask_rgb)

				unique_colors = np.unique(img_mask_rgb.reshape(-1, img_mask_rgb.shape[2]), axis=0)
				print(f"Уникальные цвета в маске:\n{unique_colors}")

				if i % 100 == 0:
					print(f"Собрано {i}/{num_samples}...")

		finally:
			self.client.simPause(False)

		print("Сбор данных завершен!")