import json

SEARCH_GRIDS = [
    # 1. Эксперименты для UNet
    {
        "SEG_MODEL_NAME": ["Unet"],
        "SEG_BACKBONE": ["resnet34", "mobilenet_v2"],
        "BATCH_SIZE": [8],                        
        "LEARNING_RATE": [1e-4, 3e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [1],
		"AUG_STRATEGY": ["none", "medium", "heavy"],
        "EXTRA_KWARGS": [
            json.dumps({}),
            json.dumps({"decoder_attention_type": "scse"})
        ]
    },
    # 2. Эксперименты для DeepLabV3Plus
    {
        "SEG_MODEL_NAME": ["DeepLabV3Plus"],
        "SEG_BACKBONE": ["resnet34", "mit_b0"],  
        "BATCH_SIZE": [4],                        
        "LEARNING_RATE": [1e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [1],
		"AUG_STRATEGY": ["medium"],
        "EXTRA_KWARGS": [json.dumps({})]
    }
]