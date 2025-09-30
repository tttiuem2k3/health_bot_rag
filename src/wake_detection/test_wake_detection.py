import wave
import struct
import pvporcupine
from src import config
# ====== Nhập access_key từ Picovoice ======
ACCESS_KEY = config.pvporcupine_key

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=config.keyword_paths,
    sensitivities=[0.9, 0.7]
)

def detect_wake_word():
    with wave.open(config.AUDIO_DIRS["esp32_record_audio"], "rb") as wf:
        n_frames = wf.getnframes()
        raw_audio = wf.readframes(n_frames)

    # Convert bytes to list of int16
    pcm = struct.unpack("<" + "h" * (len(raw_audio) // 2), raw_audio)

    for i in range(0, len(pcm) - porcupine.frame_length, porcupine.frame_length):
        frame = pcm[i:i + porcupine.frame_length]
        result = porcupine.process(frame)
        if result == 0:
            print("\n ☑️  Kiểm tra ----------> ✅ Wake word 'Hey Lisa' detected - Khởi động AI hỏi đáp bệnh tật!\n")
            return 1
        elif result == 1:
            print("\n ☑️  Kiểm tra ----------> ✅ Wake word 'Hey David' detected - Khởi động AI google assitan!\n")
            return 2

    print("\n ☑️  Kiểm tra ----------> ❌ No wake word detected - Tiếp tục lắng nghe!\n")
    return 0