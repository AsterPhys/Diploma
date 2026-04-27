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
		from train import train_model
		print("[MAIN] Запуск обучения нейросети...")
		train_model()
		
	elif args.mode == 'land':
		print("[MAIN] Режим посадки пока не реализован.")
		# TODO: import landing_logic

if __name__ == "__main__":
	main()