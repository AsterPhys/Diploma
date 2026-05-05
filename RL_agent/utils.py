import json
import airsim
import time
import pymap3d as pm
import os
import numpy as np

import config

def generate_config(start_unreal_callback, kill_unreal_callback):
    print("\n=== ГЕНЕРАЦИЯ КОНФИГА МАРШРУТОВ ===")
    if not os.path.exists(config.RAW_ROUTES_PATH):
        raise FileNotFoundError(f"Файл {config.RAW_ROUTES_PATH} не найден!")

    with open(config.RAW_ROUTES_PATH, 'r') as f:
        raw_routes = json.load(f)

    final_config = {}

    # Проходимся по всем уровням из конфига
    for level_key, level_data in raw_routes.items():
        map_name = level_data["name"]
        coords_list = level_data["coords"]
        
        # Если маршрутов нет, пропускаем уровень
        if not coords_list:
            final_config[level_key] = []
            continue

        print(f"\n[{level_key}] Подготовка карты: {map_name}...")
        
        start_unreal_callback(map_name)
        print("Подключение к AirSim...")
        client = airsim.MultirotorClient()

        # Проверяем состояние подключения
        connected = False
        n_tries = 10
        for attempt in range(n_tries):
            try:
                client.confirmConnection()
                connected = True
                break
            except Exception:
                print(f"  Ожидание движка Unreal Engine (попытка {attempt+1}/{n_tries})...")
                time.sleep(3)

        if not connected:
            raise RuntimeError(f"ВНИМАНИЕ: Не удалось подключиться к AirSim на карте {map_name}")

        # Получаем OriginGeopoint для текущей карты
        drone_state = client.getMultirotorState()
        geo_point = drone_state.gps_location
        lat0 = geo_point.latitude
        lon0 = geo_point.longitude
        alt0 = geo_point.altitude
        
        print(f"Origin (0,0,0) карты {map_name} находится в: Lat: {lat0:.6f}, Lon: {lon0:.6f}, Alt: {alt0:.2f}")

        level_routes = []
        for i, coord in enumerate(coords_list):
            start_ned = coord["start"]
            finish_ned = coord["finish"]

            start_ned = [t / 100 for t in start_ned]
            start_ned[2] *= -1
            finish_ned = [t / 100 for t in finish_ned]
            finish_ned[2] *= -1

            # Переводим локальные координаты финиша (NED) в мировые (GPS)
            target_lat, target_lon, target_alt = pm.ned2geodetic(
                finish_ned[0], finish_ned[1], finish_ned[2],
                lat0, lon0, alt0
            )

            # Логика динамического таймаута (длины маршрута)
            # в зависимости от расстояния между стартом и финишем.
            distance = np.linalg.norm(np.array(finish_ned) - np.array(start_ned))

            ideal_time = distance / config.MAX_SPEED
            ideal_steps = ideal_time / config.MOVE_TIME

            max_steps_for_route = int(ideal_steps * config.COEFF_ROUTE_STEPS) + config.BASE_ROUTE_STEPS

            # Формируем словарь в том виде, который ждет env.py
            route_dict = {
                "map_name": map_name,
                "start_local": start_ned,
                "target_gps": [target_lat, target_lon, target_alt],
                "max_steps": max_steps_for_route
            }
            level_routes.append(route_dict)
            
        final_config[level_key] = level_routes
        kill_unreal_callback()

    with open(config.ROUTES_CONFIG_PATH, 'w') as f:
        json.dump(final_config, f, indent=4)
        
    print(f"\nУспех! Файл {config.ROUTES_CONFIG_PATH} успешно сгенерирован.")