import os
import json
import sys

SETTINGS_PATH = os.path.expanduser(r"~\Documents\AirSim\settings.json")

def get_preset(mode):
	"""Возвращает словарь настроек в зависимости от режима"""
	
	# Базовые настройки, которые общие для всех режимов
	base_settings = {
		"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
		"SettingsVersion": 1.2,
		"SimMode": "Multirotor",
		"OriginGeopoint": {
			"Latitude": 55.717939,
			"Longitude": 37.795511,
			"Altitude": 260.0
		}
	}

	if mode in ["rl_train", "rl_test"]:
		view_mode = "NoDisplay" if mode == "rl_train" else "SpringArmChase"
		clock_speed = 2.0 if mode == "rl_train" else 1.0

		base_settings.update({
			"ViewMode": view_mode,
			"ClockSpeed": clock_speed,
			"CameraDefaults": {
				"CaptureSettings":[{"ImageType": 1, "Width": 84, "Height": 84, "FOV_Degrees": 90}]
			},
			"SubWindows":[
				{"WindowID": 0, "CameraName": "0", "ImageType": 1, "Visible": False}
			]
		})
	elif mode in ["collect", "land"]:
		view_mode = "NoDisplay" if mode == "collect" else "SpringArmChase"
		clock_speed = 5.0 if mode == "collect" else 1.0

		base_settings.update({
			"SegmentationSettings": {
				"InitMethod": "None",
				"OverrideExisting": False
			},
			"ViewMode": view_mode,
			"ClockSpeed": clock_speed,
			"Vehicles": {
				"SimpleFlight": {
					"VehicleType": "SimpleFlight",
					"Cameras": {
						"bottom_center": {
							"X": 0, "Y": 0, "Z": 0.2, # немного смещаем камеру вниз от центра дрона
							"Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0,
							"CaptureSettings":[
								{ "ImageType": 0, "Width": 640, "Height": 480, "FOV_Degrees": 90 }, # RGB
								{ "ImageType": 1, "Width": 640, "Height": 480, "FOV_Degrees": 90 }, # Depth
								{ "ImageType": 5, "Width": 640, "Height": 480, "FOV_Degrees": 90 }  # Mask
							]
						}
					}
				}
			}
		})
	elif mode == "pipeline":
		base_settings.update({
			"ViewMode": "SpringArmChase",
			"ClockSpeed": 2.0,
			"Vehicles": {
				"SimpleFlight": {
					"VehicleType": "SimpleFlight",
					"Cameras": {
						# Камера для RL: сдвинута чуть вперед (X=0.2), смотрит прямо (Pitch=0)
						"front_center": {
							"X": 0.2, "Y": 0.0, "Z": 0.0, 
							"Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0,
							"CaptureSettings":[
								{ "ImageType": 0, "Width": 640, "Height": 480, "FOV_Degrees": 90 }
							]
						},
						# Камера для Посадки: сдвинута чуть вниз (Z=0.2), смотрит строго вниз (Pitch=-90)
						"bottom_center": {
							"X": 0.0, "Y": 0.0, "Z": 0.2, 
							"Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0,
							"CaptureSettings":[
								{ "ImageType": 0, "Width": 640, "Height": 480, "FOV_Degrees": 90 }
							]
						}
					}
				}
			}
		})

	return base_settings

def ensure_settings(mode):
	"""Проверяет настройки и обновляет их, если нужно"""
	desired_settings = get_preset(mode)
	current_settings = {}

	if os.path.exists(SETTINGS_PATH):
		try:
			with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
				current_settings = json.load(f)
		except json.JSONDecodeError:
			print("[Warning] settings.json поврежден, он будет перезаписан.")

	# Сравниваем словари
	if current_settings != desired_settings:
		print("=" * 60)
		print(f"Обнаружены неактуальные настройки для режима '{mode}'.")
		print(f"Обновляю файл: {SETTINGS_PATH}")
		
		# Записываем новые настройки
		os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
		with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
			json.dump(desired_settings, f, indent=4)
			
		print("Файл settings.json успешно обновлен.")
		print("=" * 60)
		return True

	print(f"Настройки для '{mode}' актуальны.")
	return False