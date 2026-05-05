import json

SEARCH_GRIDS = [
    {
        "SEG_MODEL_NAME": ["Unet"],
        "SEG_BACKBONE": ["resnet34", "mobilenet_v2"],
        "BATCH_SIZE": [8],                        
        "LEARNING_RATE": [1e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [10],
		"AUG_STRATEGY": ["none", "medium", "heavy"],
        "EXTRA_KWARGS": [
            json.dumps({}),
            json.dumps({"decoder_attention_type": "scse"})
        ]
    },
    {
        "SEG_MODEL_NAME": ["DeepLabV3Plus"],
        "SEG_BACKBONE": ["resnet34", "mit_b0"],  
        "BATCH_SIZE": [4],                        
        "LEARNING_RATE": [1e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [10],
		"AUG_STRATEGY": ["medium"],
        "EXTRA_KWARGS": [json.dumps({})]
    },
	{
        "SEG_MODEL_NAME": ["LinkNet", "FPN"],
        "SEG_BACKBONE": ["mobilenet_v3_small", "efficientnet-b0"],
        "BATCH_SIZE": [16],                   
        "LEARNING_RATE": [3e-4, 5e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [10],
        "AUG_STRATEGY": ["light", "medium"],
        "EXTRA_KWARGS": [json.dumps({})]
    },
	{
        "SEG_MODEL_NAME": ["PAN"],
        "SEG_BACKBONE": ["resnet18", "efficientnet-b0"],  
        "BATCH_SIZE": [8],                        
        "LEARNING_RATE": [1e-4, 3e-4],
        "OPTIMIZER": ["AdamW"],
        "EPOCHS": [30],
        "AUG_STRATEGY": ["medium"],
        "EXTRA_KWARGS": [json.dumps({})]
    },
	{
        "SEG_MODEL_NAME":["yolo11n-seg.pt", "yolo11s-seg.pt"], 
        "SEG_BACKBONE": ["none"], # У YOLO встроенный бэкбон
        "BATCH_SIZE": [16, 32],                        
        "LEARNING_RATE": [1e-3, 3e-4],
        "OPTIMIZER": ["AdamW", "auto"],
        "EPOCHS": [30],
        "AUG_STRATEGY": ["light", "medium"],
        "EXTRA_KWARGS": [json.dumps({})]
    }
]