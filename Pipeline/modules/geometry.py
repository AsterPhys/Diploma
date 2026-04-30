import numpy as np
import cv2
import config_pipeline as cfg

class LandingMath:
	def compute_normals_and_slope(self, depth_map_metric, fov_deg=90.0):
		h, w = depth_map_metric.shape
		f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
		
		# Градиенты метрической глубины по осям X и Y
		depth_map_metric = cv2.GaussianBlur(depth_map_metric, (5, 5), 0)
		dzdx = cv2.Sobel(depth_map_metric, cv2.CV_64F, 1, 0, ksize=3) / 8.0
		dzdy = cv2.Sobel(depth_map_metric, cv2.CV_64F, 0, 1, ksize=3) / 8.0
		
		dzdx_world = dzdx * (f / np.maximum(depth_map_metric, 0.1))
		dzdy_world = dzdy * (f / np.maximum(depth_map_metric, 0.1))

		# Уклон (угол с вертикалью)
		slope_rad = np.arccos(1.0 / np.sqrt(dzdx_world**2 + dzdy_world**2 + 1.0))
		return np.degrees(slope_rad)

	def find_best_landing_spot(self, semantic_mask, depth_map, max_slope_deg):
		slope_map = self.compute_normals_and_slope(depth_map, cfg.CAMERA_FOV)
		slope_mask = (slope_map <= max_slope_deg).astype(np.uint8)
		
		# Класс 0 - Safe Ground
		safe_semantic = (semantic_mask == 0).astype(np.uint8) 

		combined_mask = cv2.bitwise_and(safe_semantic, slope_mask)
		
		if np.sum(combined_mask) < 50:
			return -1, -1, False

		dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
		_, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
		
		# Проверяем, достаточно ли велик круг для безопасной посадки
		if max_val < cfg.MIN_SAFE_ZONE_RADIUS:
			return -1, -1, False
			
		return max_loc[0], max_loc[1], True # U, V