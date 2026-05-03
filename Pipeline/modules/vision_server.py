import sys
import os

# === НАСТРОЙКИ И ФИКСЫ ДЛЯ AMD ROCm ===
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

import cv2
import json
import numpy as np
import base64
from fastapi import FastAPI, Request
import uvicorn
import segmentation_models_pytorch as smp
from PIL import Image

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PIPELINE_DIR)
import config_pipeline as config

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Сегментация ===
print(f"[SERVER] Чтение настроек сегментации из: {config.SEG_CONFIG_PATH}")
with open(config.SEG_CONFIG_PATH, 'r', encoding='utf-8') as f:
	seg_cfg = json.load(f)

# Получаем параметры, с которыми обучалась модель
model_arch = seg_cfg["model"]["architecture"]
model_backbone = seg_cfg["model"]["backbone"]
num_classes = seg_cfg["model"]["num_classes"]

print(f"[SERVER] Инициализация: {model_arch} | Backbone: {model_backbone} | Классов: {num_classes}")

model_class = getattr(smp, model_arch)
seg_model = model_class(
	encoder_name=model_backbone,
	encoder_weights=None,
	in_channels=3,
	classes=num_classes
)

# Загружаем веса
print(f"[SERVER] Загрузка весов из: {config.SEG_MODEL_PATH}")
checkpoint = torch.load(config.SEG_MODEL_PATH, map_location=device)
seg_model.load_state_dict(checkpoint['model_state_dict'])
seg_model.to(device).eval()

# === ENDPOINT ===
@app.post("/predict")
async def predict(request: Request):
	data = await request.json()
	
	img_bytes = base64.b64decode(data['image'])
	np_img = np.frombuffer(img_bytes, dtype=np.uint8)
	frame_rgb = cv2.cvtColor(cv2.imdecode(np_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
	
	camera_type = data.get("camera", "front") 
	mask_b64, depth_b64 = "", ""

	with torch.no_grad():
		if camera_type == "bottom":
			img_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
			mask = torch.argmax(seg_model(img_t), dim=1).cpu().numpy()[0].astype(np.uint8)
			mask_b64 = base64.b64encode(mask.tobytes()).decode('utf-8')

	return {
		"mask_b64": mask_b64,
		"shape": [frame_rgb.shape[0], frame_rgb.shape[1]]
	}

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)