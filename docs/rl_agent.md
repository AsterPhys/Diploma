```markdown
# Модуль RL_agent
## 1. Назначение модуля
`RL_agent` реализует обучение агента автономной навигации в AirSim на основе многомодального наблюдения:
- стек depth-кадров (пространственно-временной контекст),
- вектор до цели в body frame (направление + нормализованная дистанция).
Поддерживаются алгоритмы `PPO` и `SAC` (`config.py`), приоритетно используется curriculum-подход с поэтапным усложнением маршрутов и уровней.
## 2. Связка `manager.py -> train.py -> env.py`
### `manager.py` (оркестрация процесса)
- вызывает `ensure_settings("rl_train")`;
- при необходимости генерирует `routes_config.json` через `utils.generate_config(...)`;
- определяет текущую карту по `state.json` и запускает UE именно на нужном уровне;
- запускает `train.py` как подпроцесс;
- анализирует код возврата:
  - `0` — штатное завершение;
  - `42` — сигнал смены уровня (перезапуск UE на новой карте);
  - иначе — аварийный рестарт цикла.
### `train.py` (алгоритм и логика обучения)
- поднимает watchdog-поток (защита от зависаний UE/step-loop);
- создает среду `ColosseumDroneEnv`, оборачивает в `Monitor` и `DummyVecEnv`;
- строит мультимодальный extractor:
  - CNN-ветка для depth,
  - MLP-ветка для вектора,
  - fusion в общий embedding;
- инициализирует PPO/SAC с параметрами из `config.py`;
- поддерживает resume:
  - `latest_model.zip`,
  - `state.json`,
  - `replay_buffer.pkl` (для SAC);
- запускает `CurriculumCallback`, который:
  - ведет статистику успехов по маршрутам,
  - открывает новые маршруты,
  - повышает уровень,
  - сохраняет прогресс.
### `env.py` (Gymnasium-среда)
- подключается к AirSim и читает `routes_config.json`;
- на `reset()` выбирает маршрут из пула разблокированных;
- формирует наблюдение:
  - depth с шумами и dropout (sim-to-real),
  - вектор до цели в системе дрона;
- на `step(action)`:
  - преобразует действие из body frame в world frame,
  - отправляет команду в AirSim,
  - считает reward:
    - прогресс к цели,
    - шаговый штраф,
    - бонус за успех,
    - сильный штраф за collision;
  - завершает эпизод по успеху/столкновению/таймауту маршрута.
## 3. Curriculum Learning: логика смены уровней
Curriculum реализован в `CurriculumCallback`:
- ведутся буферы успешности по каждому маршруту (`deque` фиксированной длины);
- через заданный интервал шагов проверяются условия:
  1. **unlock route по quality**: если winrate по всем открытым маршрутам >= порога;
  2. **unlock route по timeout**: если слишком долго нет прогресса;
  3. **level up по mastery**: если все маршруты уровня освоены;
  4. **level up по timeout**: если превышен лимит шагов уровня.
- при смене уровня вызывается `env.set_level(new_lvl)` и процесс завершает текущий запуск кодом `42`, чтобы `manager.py` перезапустил UE на новой карте.
Таким образом, переходы между уровнями синхронизированы с инфраструктурой симулятора и не завязаны на «горячую» замену карты внутри одного процесса.
## 4. Диаграмма жизненного цикла среды
```mermaid
flowchart TD
    A[reset()] --> B[Выбор маршрута из unlocked пулa]
    B --> C[Телепорт в старт + прогрев камеры]
    C --> D[Формирование obs: depth_stack + vector]
    D --> E[step(action)]
    E --> F[Body->World transform + move]
    F --> G[Новое obs + reward]
    G --> H{Терминальные условия}
    H -- успех --> I[terminated=True, bonus]
    H -- collision --> J[terminated=True, penalty]
    H -- timeout --> K[truncated=True]
    H -- иначе --> E
```

## 5. Диаграмма переходов по уровням (Curriculum)

```stateDiagram-v2
    [*] --> LevelN: старт/восстановление state

    state LevelN {
        [*] --> TrainRoutes
        TrainRoutes --> TrainRoutes: сбор статистики winrate
        TrainRoutes --> UnlockRoute: winrate >= threshold\nдля всех открытых
        TrainRoutes --> UnlockRoute: timeout unlock
        UnlockRoute --> TrainRoutes: opened next route
    }

    LevelN --> LevelNplus1: all routes passed
    LevelN --> LevelNplus1: max steps per level exceeded
    LevelNplus1 --> [*]: exit code 42 -> manager restart UE
```

## 6. Архитектурный вывод

`RL_agent` сочетает алгоритмический контур RL и инфраструктурный контур управления симулятором. Благодаря этому обучение устойчиво к зависаниям и масштабируется по сложности среды через формализованный curriculum-механизм.