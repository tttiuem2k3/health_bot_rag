from dotenv import load_dotenv
import os

# Load biến môi trường từ file .env
load_dotenv()

# Lấy API keys
api_keys_llama = [os.getenv("LLAMA_KEYS")]
api_keys_gemini = [os.getenv("GEMINI_KEYS")]
pvporcupine_key = os.getenv("PVKEY")

# Đường dẫn wake word detection 'keyword_paths'
keyword_paths = [
    "./model/wake_word/Hey-lisa_en_windows_v3_0_0.ppn",
    "./model/wake_word/Hey-David_en_windows_v3_0_0.ppn"
]
model_vosk_small= "./model/vosk/vosk-model-small-vn-0.4"
model_vosk= "./model/vosk/vosk-model-vn-0.4"

# Đường dẫn thư mục của hệ thống iot
AUDIO_DIRS = {
    "music": "./src/iot/music",
    "alarm": "./src/iot/alarm",
    "audio_wake_respon": "./src/iot/audio_wake_respon",
    "chunks": "./src/iot/stream_chunks",
    "esp32_record_audio": "./src/iot/combined_audio.wav",
    "ai_output_audio": "./src/iot/output.wav",
    "output_audio_to_esp32": "./src/iot/combined_audio_esp32.wav"
}
# Đường dẫn thư mục chứa dữ liệu Document về bệnh tật
Data_health_path = "./data_source/health_bot"

# Đường dẫn thư mục chứa dữ liệu Document khác
Data_orther_path = "./data_source/machine_learning"

# Đường dẫn thư mục chứa dữ liệu Document về bệnh tật đã được embedding
Data_health_embedding_path = "./data_source/health_bot/chroma_db1"
# Đường dẫn đến thư mục chứa dữ liệu tên bệnh
disease_list_file = "./data_source/health_bot/disease_list.txt"