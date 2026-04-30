import requests
import cv2
import numpy as np
import base64

class VisionPipeline:
	def __init__(self, server_url):
		self.url = server_url
		print(f"[VISION] Клиент подключен к серверу: {self.url}")

	def get_predictions(self, frame_bgr, camera="front"):
		"""Базовый метод отправки кадра на сервер."""
		_, buffer = cv2.imencode('.jpg', frame_bgr)
		img_base64 = base64.b64encode(buffer).decode('utf-8')

		payload = {"image": img_base64, "camera": camera}

		try:
			response = requests.post(self.url, json=payload, timeout=30.0)
			result = response.json()
			
			h, w = result["shape"]
			mask, depth = None, None

			# Глубина (нужна для обоих режимов)
			depth_bytes = base64.b64decode(result['depth_b64'])
			depth = np.frombuffer(depth_bytes, dtype=np.float16).reshape((h, w)).astype(np.float32)

			# Маска (нужна только при посадке)
			if result.get('mask_b64'):
				mask_bytes = base64.b64decode(result['mask_b64'])
				mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w))

			return mask, depth

		except Exception as e:
			print(f"[VISION ERROR] Ошибка связи с сервером ИИ: {e}")
			h, w = frame_bgr.shape[:2]

			# Если сервер упал, возвращаем опасные значения, 
			# чтобы дрон не отключил моторы в воздухе :/

			safe_mask = np.full((h, w), 255, dtype=np.uint8)
			safe_depth = np.full((h, w), 20.0, dtype=np.float32)

			return safe_mask, safe_depth

	def predict_depth(self, frame_bgr):
		"""Для передней камеры (навигация RL). Возвращает только карту глубины."""
		_, depth = self.get_predictions(frame_bgr, camera="front")
		return depth

	def predict_landing_data(self, frame_bgr):
		"""Для нижней камеры (посадка). Возвращает и маску, и глубину."""
		mask, depth = self.get_predictions(frame_bgr, camera="bottom")
		return mask, depth