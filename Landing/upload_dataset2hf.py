import time
import os
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "asterphys/airsim-drone-dataset"
FOLDER_PATH = "./dataset"

api = HfApi()

print(f"[*] Начинаю загрузку через upload_large_folder...")
print(f"[*] Репозиторий: {REPO_ID}")

while True:
    try:
        api.upload_large_folder(
            folder_path=FOLDER_PATH,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        
        print("\n[SUCCESS] Датасет успешно загружен!")
        break

    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            print(f"\n[!] Лимит Hugging Face (429). Сервер перегружен запросами.")
            # Читаем из ответа сервера, сколько именно надо ждать.
            # Если нет - спим стандартно 15 минут.
            wait_time = 900 
            print(f"[#] Уходим в режим ожидания на {wait_time//60} минут...")
            time.sleep(wait_time)
        else:
            print(f"\n[!] Произошла ошибка HTTP: {e}")
            print("Пробую снова через 60 секунд...")
            time.sleep(60)
            
    except Exception as e:
        print(f"\n[!] Непредвиденная ошибка: {e}")
        print("Пробую снова через 30 секунд...")
        time.sleep(30)