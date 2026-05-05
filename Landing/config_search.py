import json

# Baseline: модель (Unet+resnet34) + spatial аугментации

SEARCH_GRIDS = [
    # ==========================================
    # ИССЛЕДОВАНИЕ 1: Влияние аугментаций (Ablation on Augmentations)
    # Baseline модель прогоняется с разными аугментациями.
    # ==========================================
    {
        "SEG_MODEL_NAME": ["Unet"],
        "SEG_BACKBONE": ["resnet34"],
        "BATCH_SIZE": [16],                        
        "LEARNING_RATE": [3e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [25],
        "AUG_STRATEGY": ["none", "spatial", "lighting", "sensor"],
        "EXTRA_KWARGS": [json.dumps({})]
    },

    # ==========================================
    # ИССЛЕДОВАНИЕ 2: Выбор оптимального энкодера (Backbone Search)
    # Фиксируем модель (Unet) и аугментации (medium), меняем только backbone.
    # ==========================================
    {
        "SEG_MODEL_NAME": ["Unet"],
        "SEG_BACKBONE": ["mobilenet_v3_small", "efficientnet-b0", "resnet18"],
        "BATCH_SIZE": [16],                        
        "LEARNING_RATE": [3e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [25],
        "AUG_STRATEGY": ["spatial"],
        "EXTRA_KWARGS": [json.dumps({})]
    },

    # ==========================================
    # ИССЛЕДОВАНИЕ 3: Выбор архитектуры декодера (Architecture Search)
    # Фиксируем бэкбон (resnet34) и аугментации (lighting), меняем head.
    # ==========================================
    {
        "SEG_MODEL_NAME": ["FPN", "PAN", "LinkNet", "DeepLabV3Plus"],
        "SEG_BACKBONE": ["resnet34"],  
        "BATCH_SIZE": [16],                        
        "LEARNING_RATE": [3e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [25],
        "AUG_STRATEGY": ["spatial"],
        "EXTRA_KWARGS": [json.dumps({})]
    },

    # ==========================================
    # ИССЛЕДОВАНИЕ 4: Влияние Attention-механизмов
    # Берем наш Baseline и добавляем Spatial-Channel Attention (scSE).
    # Помогает ли механизм внимания лучше видеть мелкие препятствия (провода, столбы)?
    # ==========================================
    {
        "SEG_MODEL_NAME": ["Unet"],
        "SEG_BACKBONE": ["resnet34"],
        "BATCH_SIZE": [16],                        
        "LEARNING_RATE": [3e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS":[25],
        "AUG_STRATEGY": ["spatial"],
        "EXTRA_KWARGS": [json.dumps({"decoder_attention_type": "scse"})]
    },

    # ==========================================
    # ИССЛЕДОВАНИЕ 5: Сравнение с SOTA (YOLOv11)
    # Сравниваем классический подход (SMP) с передовым (YOLO).
    # ==========================================
    {
        "SEG_MODEL_NAME": ["yolo11n-seg.pt", "yolo11s-seg.pt"], 
        "SEG_BACKBONE": ["none"],
        "BATCH_SIZE": [16],                        
        "LEARNING_RATE": [1e-3],
        "OPTIMIZER": ["auto"],
        "EPOCHS": [30],
        "AUG_STRATEGY": ["none", "spatial", "lighting", "yolo_advanced"],
        "EXTRA_KWARGS": [json.dumps({})]
    }
]