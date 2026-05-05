```markdown
# Модуль Landing

## 1. Назначение модуля

`Landing` реализует полный ML-контур семантической посадки:
1. генерация синтетического датасета в AirSim (`collector.py`);
2. обучение модели сегментации (`train.py`);
3. управление сценариями запуска и перезапуска (`manager.py`);
4. подбор гиперпараметров через grid search (`config_search.py` + режим `search`).

Цель модуля — построить модель, выделяющую безопасные и опасные классы поверхности для дальнейшего использования в боевом `Pipeline`.

## 2. Логика ключевых файлов

### `collector.py`
Основной класс `DataCollector`:
- подключается к AirSim и включает погодные эффекты;
- генерирует сетку спавна внутри полигонов (`SPAWN_POLYGON`, `HOTZONE_POLYGON`);
- выполняет randomization (время суток, туман/пыль);
- телепортирует дрон в точки сетки с jitter;
- собирает три синхронных канала: RGB, Depth, Segmentation;
- фильтрует некачественные кадры:
  - коллизия при спавне,
  - слишком темный кадр,
  - дубликаты (буферный лаг),
  - аномальная глубина,
  - наличие неразмеченных (черных) пикселей;
- конвертирует цветовую маску AirSim в целочисленные классы и сохраняет набор артефактов (`rgb`, `depth`, `mask`, `depth_vis`, `mask_vis`).

### `train.py`
Скрипт обучения сегментации:
- кастомный `DroneLandingDataset` с ремаппингом классов (в т.ч. объединение `Hazard` со статическими препятствиями);
- аугментации из `config_train.py` (`none/light/medium/heavy`);
- архитектуры из `segmentation_models_pytorch` (`Unet`, `DeepLabV3Plus` и др.);
- mixed precision (`torch.amp`), TensorBoard-логирование;
- метрики: per-class IoU, mIoU, отдельный бинарный показатель `any_obstacle`;
- авто-resume по последнему `model_epoch_*.pth`;
- чекпойнтинг каждой эпохи + `best_model.pth` по `IoU Safe_Ground`.

### Конфигурации

- `config.py`: инфраструктура и сбор данных (пути, карта UE, полигоны спавна, камера, randomization).
- `config_train.py`: параметры обучения, аугментации, class map, сериализация train-конфига.
- `config_search.py`: сетки гиперпараметров для пакетного перебора экспериментов.
- `manager.py`: точка входа для `collect/train/search`, контроль Unreal, формирование `RUN_NAME`, возобновление запусков.

## 3. Диаграмма пайплайна Landing

```mermaid
flowchart TD
    A[manager.py --mode collect] --> B[collector.py]
    B --> C[Генерация точки спавна<br/>Polygon + Grid + Jitter]
    C --> D[Randomize weather/time]
    D --> E[Capture RGB + Depth + Segmentation]
    E --> F{QC фильтры<br/>collision/dark/duplicate/unlabeled/depth}
    F -- fail --> C
    F -- pass --> G[Сохранение data/*]

    G --> H[manager.py --mode train/search]
    H --> I[train.py]
    I --> J[Dataset + Augmentations]
    J --> K[SMP model training]
    K --> L[IoU/mIoU + TensorBoard]
    L --> M[Checkpoint + best_model.pth + config.json]
	```

	## 4. Архитектурные особенности

	- Сбор данных ориентирован на покрытие пространства (grid + hotzone bias), а не только случайный sampling.
	- Контроль качества встроен в online-цикл коллектора, что снижает шум в датасете до этапа обучения.
	- Эксперименты воспроизводимы: конфигурация каждого запуска сохраняется в run-директории.