import time
import config_pipeline as cfg
from enum import Enum
import sys
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.append(ROOT_DIR)

RL_DIR = os.path.join(ROOT_DIR, "RL_agent")
sys.path.append(RL_DIR)
SS_DIR = os.path.join(ROOT_DIR, "Landing")
sys.path.append(SS_DIR)

# --- ПОДКЛЮЧАЕМ МЕНЕДЖЕР НАСТРОЕК ---
from settings_manager import ensure_settings

settings_changed = ensure_settings("pipeline")
if settings_changed:
    print("\n" + "!" * 60)
    print("ВНИМАНИЕ: Настройки AirSim обновлены на режим 'pipeline'.")
    print("!" * 60 + "\n")
    time.sleep(20)

from modules.control import DroneController
from modules.rl_wrapper import RLNavigator
from modules.vision import VisionPipeline
from modules.geometry import LandingMath

class DroneState(Enum):
	TAKEOFF = 1
	NAVIGATE_RL = 2
	AUTO_LANDING = 3
	DONE = 4

class LandingStateMachine:
	def __init__(self):
		print("[INIT] Запуск систем...")
		self.drone = DroneController()
		self.vision = VisionPipeline(cfg.VISION_SERVER_URL)
		self.geom = LandingMath()
		
		# Рассчитываем цель один раз на старте
		self.target_ned = self.drone.ue_to_ned(cfg.START_UE_COORDS, cfg.FINISH_UE_COORDS)
		self.initial_dist = self.drone.get_distance_to_target(self.target_ned)
		
		t_lat, t_lon, t_alt = self.drone.get_target_gps(self.target_ned)
		print(f"[INIT] Цель в локальных метрах (NED): X={self.target_ned[0]:.1f}, Y={self.target_ned[1]:.1f}")
		print(f"[INIT] Цель в GPS: Latitude {t_lat:.6f}, Longitude {t_lon:.6f}, Altitude {t_alt:.2f}m")

		self.rl_agent = RLNavigator(cfg.RL_MODEL_PATH, initial_distance=self.initial_dist)
		self.state = DroneState.TAKEOFF

	def run(self):
		try:
			while self.state != DroneState.DONE:
				if self.state == DroneState.TAKEOFF:
					self._state_takeoff()
				elif self.state == DroneState.NAVIGATE_RL:
					self._state_navigate()
				elif self.state == DroneState.AUTO_LANDING:
					self._state_landing()
		except KeyboardInterrupt:
			print("Прервано пользователем. Остановка...")
			self.drone.stop_moving()

	def _state_takeoff(self):
		self.drone.arm_and_takeoff(cfg.TAKEOFF_ALTITUDE)
		self.state = DroneState.NAVIGATE_RL
		print("[STATE] Взлет завершен. Переход к навигации RL.")

	def _state_navigate(self):
		# Запрашиваем только глубину с передней камеры
		front_rgb = self.drone.get_camera_image("front_center")
		depth_map = self.vision.predict_depth(front_rgb)
		
		drone_state = self.drone.client.getMultirotorState()
		pos = drone_state.kinematics_estimated.position
		orient = drone_state.kinematics_estimated.orientation
		
		dist = self.drone.get_distance_to_target(self.target_ned)

		if dist < cfg.RL_ARRIVAL_DISTANCE:
			print(f"[STATE] Цель достигнута ({dist:.2f}м). Переход к автопосадке.")
			self.drone.stop_moving()
			self.state = DroneState.AUTO_LANDING
			return

		vector = self.rl_agent.get_body_frame_vector(pos, orient, self.target_ned)
		action = self.rl_agent.predict(depth_map, vector)
		
		self.drone.move_by_action(action)

	def _state_landing(self):
		bottom_rgb = self.drone.get_camera_image("bottom_center")
		semantic_mask, depth_map = self.vision.predict_landing_data(bottom_rgb)
		
		target_u, target_v, is_valid = self.geom.find_best_landing_spot(semantic_mask, depth_map, cfg.MAX_SLOPE_DEGREES)
		
		if not is_valid:
			print("[Посадка] Безопасная зона не найдена! Ищу...")
			# Если зона не найдена, летим медленно вверх и вперед, чтобы изменить обзор
			self.drone.move_velocity_ned(0.5, 0.0, -0.2) 
			time.sleep(0.1)
			return

		vx, vy = self.drone.pid_compute_velocity(target_u, target_v, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, 
												 cfg.PID_Kp, cfg.PID_Ki, cfg.PID_Kd)
		
		center_y, center_x = cfg.IMAGE_HEIGHT // 2, cfg.IMAGE_WIDTH // 2
		distance_to_ground = depth_map[center_y, center_x]

		if distance_to_ground < 0.4:
			print("[Посадка] Касание земли. Моторы выключены.")
			self.drone.disarm()
			self.state = DroneState.DONE

			final_gps = self.drone.client.getMultirotorState().gps_location
			print("[Посадка] Касание земли. Моторы выключены.")
			print(f"[ФИНИШ] Фактическая точка посадки GPS: Lat {final_gps.latitude:.6f}, Lon {final_gps.longitude:.6f}")
		else:
			# Снижаемся с заданной скоростью
			self.drone.move_velocity_ned(vx, vy, cfg.LANDING_SPEED_Z)
			time.sleep(0.1)

if __name__ == "__main__":
	sm = LandingStateMachine()
	sm.run()