import argparse
import config

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, choices=['collect', 'train', 'land'], required=True)
	args = parser.parse_args()

	if args.mode == 'collect':
		from collector import DataCollector
		print("[MAIN] Запуск сбора данных...")
		collector = DataCollector()
		collector.collect_data(config.DATASET_SIZE)
		
	elif args.mode == 'train':
		print("[MAIN] Запуск обучения нейросети...")
		import os
		model_name = os.environ.get("SEG_MODEL_NAME", "").lower()
		from train import train_model
		if "yolo" in model_name:
			import train_yolo
			train_yolo.main()
		else:
			from train import train_model
			train_model()

if __name__ == "__main__":
	main()