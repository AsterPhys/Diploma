import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PIPELINE_DIR)
import config_pipeline as config

import json
import torch
import numpy as np
import cv2
from collections import deque
from stable_baselines3 import PPO

class RLNavigator:
	def __init__(self, model_path, initial_distance, n_frames=4, img_size=(84, 84)):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print(f"[SERVER] Чтение настроек RL-агента из: {config.RL_CONFIG_PATH}")
		with open(config.RL_CONFIG_PATH, 'r', encoding='utf-8') as f:
			rl_cfg = json.load(f)

		# self.max_speed = rl_cfg['MAX_SPEED']
		self.max_speed = config.MAX_SPEED
		self.distance_clip_thr = rl_cfg["DISTANCE_CLIP_THR"]
		self.move_time = rl_cfg['MOVE_TIME']

		print(f"[RL] Загрузка модели: {model_path}...")
		self.model = PPO.load(model_path, device=self.device)
		self.n_frames = n_frames
		self.img_size = img_size
		self.frames_buffer = deque(maxlen=n_frames)
		self.is_buffer_ready = False
		
		# Сохраняем изначальное расстояние от старта до цели для нормализации
		self.initial_distance = initial_distance if initial_distance > 1.0 else 1.0

	def _preprocess_depth(self, depth_map):
		resized = cv2.resize(depth_map, self.img_size, interpolation=cv2.INTER_AREA)
		normalized = np.clip(resized / self.distance_clip_thr, 0.0, 1.0)
		return normalized

	def predict(self, current_depth, vector):
		processed_depth = self._preprocess_depth(current_depth)

		if not self.is_buffer_ready:
			for _ in range(self.n_frames):
				self.frames_buffer.append(processed_depth)
			self.is_buffer_ready = True
		else:
			self.frames_buffer.append(processed_depth)

		depth_stack = np.stack(self.frames_buffer, axis=0)

		obs = {
			"depth": depth_stack.astype(np.float32),
			"vector": np.array(vector, dtype=np.float32)
		}

		action, _ = self.model.predict(obs, deterministic=True)
		return action

	def get_body_frame_vector(self, drone_pos, drone_orientation, target_pos):
		from scipy.spatial.transform import Rotation as R

		target_vector_ned = np.array([
			target_pos[0] - drone_pos.x_val,
			target_pos[1] - drone_pos.y_val,
			target_pos[2] - drone_pos.z_val
		])
		dist = np.linalg.norm(target_vector_ned)

		rotation = R.from_quat([drone_orientation.x_val, drone_orientation.y_val, drone_orientation.z_val, drone_orientation.w_val])
		body_vector = rotation.inv().apply(target_vector_ned)

		normalized_vector = body_vector / dist if dist > 0 else body_vector
		normalized_dist = np.clip(dist / self.initial_distance, 0.0, 1.0)

		return [normalized_vector[0], normalized_vector[1], normalized_vector[2], normalized_dist]