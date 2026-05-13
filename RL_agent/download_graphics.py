import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def tb_logs_to_csv(log_dir, output_file="training_results.csv"):
    all_data = []

    for run_name in os.listdir(log_dir):
        run_path = os.path.join(log_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()

        tags = ea.Tags()['scalars']
        
        for tag in tags:
            # Если нужно выгрузить только конкретные метрики, добавьте фильтр тут
            # if 'rollout/ep_rew_mean' not in tag: continue

            events = ea.Scalars(tag)
            for event in events:
                all_data.append({
                    'run': run_name,
                    'step': event.step,
                    'time': event.wall_time,
                    'metric': tag,
                    'value': event.value
                })

    df = pd.DataFrame(all_data)
    
    df.to_csv(output_file, index=False)
    print(f"Готово! Данные сохранены в {output_file}")
    return df

df = tb_logs_to_csv('rl_tensorboard')