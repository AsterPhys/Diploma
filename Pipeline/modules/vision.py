import requests
import cv2
import numpy as np
import base64

class VisionPipeline:
	def __init__(self, server_url):
		self.url = server_url
		print(f"[VISION] Клиент подключен к серверу: {self.url}")

	def predict_landing_mask(self, frame_bgr, camera="bottom"):
		"""Отправка кадра на сервер для получения семантической маски."""
		_, buffer = cv2.imencode('.jpg', frame_bgr)
		img_base64 = base64.b64encode(buffer).decode('utf-8')

		payload = {"image": img_base64, "camera": camera}

		try:
			response = requests.post(self.url, json=payload, timeout=30.0)
			result = response.json()
			
			h, w = result["shape"]
			mask = None
			# Маска (нужна только при посадке)
			if result.get('mask_b64'):
				mask_bytes = base64.b64decode(result['mask_b64'])
				mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w))

			return mask

		except Exception as e:
			print(f"[VISION ERROR] Ошибка связи с сервером ИИ: {e}")
			h, w = frame_bgr.shape[:2]

			# Если сервер упал, возвращаем опасные значения, 
			# чтобы дрон не отключил моторы в воздухе :/
			safe_mask = np.full((h, w), 255, dtype=np.uint8)
			
			return safe_mask