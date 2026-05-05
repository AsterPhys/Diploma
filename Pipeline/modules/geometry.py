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
		
		h, w = dist_transform.shape
		center_v, center_u = h // 2, w // 2

		# Если под дроном (в центре кадра) уже безопасно, то просто
		# делаем центр нашей целью, чтобы дрон плавно сел вертикально вниз.
		if dist_transform[center_v, center_u] >= cfg.MIN_SAFE_ZONE_RADIUS:
			return center_u, center_v, True, combined_mask

		# Маска всех потенциально безопасных точек
		safe_mask = dist_transform >= cfg.MIN_SAFE_ZONE_RADIUS

		# Если таких точек нет вообще
		if not np.any(safe_mask):
			return -1, -1, False, combined_mask

		# Создаем сетку координат для вычисления расстояний до центра
		y_coords, x_coords = np.indices((h, w))
		dist_to_center = np.sqrt((x_coords - center_u)**2 + (y_coords - center_v)**2)

		# Взвешенная функция стоимости
		# Мы максимизируем расстояние до препятствий и минимизируем до центра.
		# Коэффициент alpha определяет баланс.
		# Score = Безопасность - (alpha * Отклонение_от_центра)
		# Чем меньше alpha, тем безопаснее выбранная точка, чем больше, тем ближе к центру.

		alpha = 0.8
		score_map = dist_transform - (alpha * dist_to_center)
		
		# Исключаем из поиска точки, которые не удовлетворяют минимальной безопасности
		score_map[~safe_mask] = -np.inf

		# Находим индекс максимального скора
		best_v, best_u = np.unravel_index(np.argmax(score_map), score_map.shape)

		return best_u, best_v, True, combined_mask