import argparse
import config
from collector import DataCollector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['collect', 'train', 'land'], required=True)
    args = parser.parse_args()

    if args.mode == 'collect':
        print("[MAIN] Запуск сбора данных...")
        collector = DataCollector()
        collector.collect_data(config.DATASET_SIZE)
        
    elif args.mode == 'train':
        print("[MAIN] Режим обучения пока не реализован.")
        # TODO: import train_pipeline
        
    elif args.mode == 'land':
        print("[MAIN] Режим посадки пока не реализован.")
        # TODO: import landing_logic

if __name__ == "__main__":
    main()