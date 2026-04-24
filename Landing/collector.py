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
		
		self.client.simEnableWeather(True)

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

		# Аналогично для hotzone
		self.hotzone = np.array(config.HOTZONE_POLYGON, dtype=np.float32)
		self.hz_min_x, self.hz_min_y = np.min(self.hotzone, axis=0)
		self.hz_max_x, self.hz_max_y = np.max(self.hotzone, axis=0)

		# Переменная для хранения предыдущего кадра
		self.prev_img_rgb = None

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
		
		use_hotzone = random.random() < config.HOTZONE_PROBABILITY
		target_poly = self.hotzone if use_hotzone else self.polygon
		min_x, max_x = (self.hz_min_x, self.hz_max_x) if use_hotzone else (self.min_x, self.max_x)
		min_y, max_y = (self.hz_min_y, self.hz_max_y) if use_hotzone else (self.min_y, self.max_y)

		while True:
			x = random.uniform(min_x, max_x)
			y = random.uniform(min_y, max_y)
			
			# Проверяем расстояние до ближайшей границы полигона.
			# Положительное число означает, что точка внутри.
			dist = cv2.pointPolygonTest(target_poly, (x, y), measureDist=True)

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

	def is_duplicate(self, current_img):
		"""Проверяет, является ли кадр дубликатом предыдущего."""
		if self.prev_img_rgb is None:
			return False
		
		# Сравниваем среднюю абсолютную разницу пикселей.
		# Если разница < 1.0, значит кадры почти идентичны.
		diff = np.mean(np.abs(current_img.astype(np.float32) - self.prev_img_rgb.astype(np.float32)))
		return diff < 1.0

	def collect_data(self, num_samples):
		print(f"Начинаем сбор {num_samples} кадров...")
		
		self.client.simPause(True)
		
		try:
			# --- WARM-UP ---
			print("[COLLECTOR] Прогрев симулятора и инициализация камеры...")
			warmup_pose = self.get_random_pose()
			warmup_pose.position.z_val = -100.0  # чтобы дрон точно не был в другом объекте
			self.client.simSetVehiclePose(warmup_pose, ignore_collision=True)
			
			if config.ENABLE_ENV_RANDOMIZATION:
				self.randomize_environment()
			
			self.client.simContinueForFrames(30)
			print("[COLLECTOR] Сброс буферов (холостой снимок)...")
			self.client.simGetImages([
				airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Scene, False, False),
				airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.DepthPlanar, True, False),
				airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Segmentation, False, False)
			])
			print("[COLLECTOR] Прогрев завершен. Начинаем запись.")

			saved_count = 0  # счетчик успешных кадров
			total_pixels = config.IMAGE_WIDTH * config.IMAGE_HEIGHT

			while saved_count < num_samples:
				# Меняем погоду каждые N кадров
				if config.ENABLE_ENV_RANDOMIZATION and saved_count % config.ENV_UPDATE_FREQUENCY == 0:
					self.randomize_environment()
				
				pose = self.get_random_pose()
				self.client.simSetVehiclePose(pose, ignore_collision=True)
				
				# Прокручиваем симуляцию
				self.client.simContinueForFrames(1)

				# Проверка коллизии (если дрон внутри меша)
				collision_info = self.client.simGetCollisionInfo()
				if collision_info.has_collided:
					print(f"[{saved_count + 1}/{num_samples}] Пропуск: спавн внутри геометрии (коллизия).")
					continue

				# Запрашиваем 3 картинки
				responses = self.client.simGetImages([
					airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Scene, False, False),
					airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.DepthPlanar, True, False),
					airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Segmentation, False, False)
				])

				# Проверка на пустые ответы от AirSim
				if any(r.image_data_uint8 is None and r.image_data_float is None for r in responses):
					continue

				# --> RGB
				img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
				img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
				
				# Проверка на то, что большая часть изображения - черная.
				black_pixels = np.sum(np.all(img_rgb == [0, 0, 0], axis=-1))
				if (black_pixels / total_pixels) > 0.70:
					print(f"[{saved_count + 1}/{num_samples}] Пропуск: большой процент черного цвета.")
					continue
				
				# Проверка на дубликат
				if self.is_duplicate(img_rgb):
					print(f"[{saved_count + 1}/{num_samples}] Пропуск: обнаружен дубликат кадра.")
					continue

				self.prev_img_rgb = img_rgb.copy()

				# --> Depth
				depth_img = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)

				# Если минимальная дистанция до объекта меньше заявленной,
				# то нам такое не надо
				if np.min(depth_img) < 0.5:
					print(f"[{saved_count + 1}/{num_samples}] Пропуск: камера в упор к текстуре/внутри меша (Depth < 0.01м).")
					continue

				# --> Segmentation
				mask1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
				img_mask_rgb = mask1d.reshape(responses[2].height, responses[2].width, 3)

				# Проверка на черный цвет на маске (неразмеченные объекты)
				has_black = np.any(np.all(img_mask_rgb == [0, 0, 0], axis=-1))
				if has_black:
					print(f"[{saved_count + 1}/{num_samples}] Пропуск: обнаружен неразмеченный черный цвет (глючный кадр).")
					continue

				unique_colors = np.unique(img_mask_rgb.reshape(-1, 3), axis=0)
				print(f"Уникальные цвета на кадре: {unique_colors.tolist()}")

				# --> Визуализация карты глубины
				# Нормализуем массив (переводим float метры в диапазон 0-255).
				depth_norm = cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
				depth_vis = np.uint8(depth_norm)
				# Применяем тепловую карту для красоты.
				# Близкие объекты будут синими, средние - зелеными/желтыми, далекие - красными.
				depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

				# --> Сохранение
				cv2.imwrite(os.path.join(config.RGB_DIR, f"{saved_count:05d}.png"), img_rgb)
				np.save(os.path.join(config.DEPTH_DIR, f"{saved_count:05d}.npy"), depth_img)
				cv2.imwrite(os.path.join(config.DEPTH_VIS_DIR, f"{saved_count:05d}.png"), depth_colormap)
				cv2.imwrite(os.path.join(config.MASK_VIS_DIR, f"{saved_count:05d}.png"), img_mask_rgb)

				saved_count += 1
				print(f"[УСПЕХ] Сохранен кадр {saved_count}/{num_samples}")
				
		finally:
			self.client.simPause(False)

		print("Сбор данных завершен!")