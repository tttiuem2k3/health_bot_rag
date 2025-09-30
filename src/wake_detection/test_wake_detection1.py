from src import config
import speech_recognition as sr

# Từ khóa cần phát hiện (có thể tùy chỉnh)
WAKE_WORDS = {
    "hi lisa": 1,
    "lisa": 1,
    "li sa": 1,
    "hi david": 2,
    "david": 2,
    "đa vít": 2
}

def detect_wake_word():
    recognizer = sr.Recognizer()
    text = ""
    with sr.AudioFile(config.AUDIO_DIRS["esp32_record_audio"]) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="vi-VN")
        except sr.UnknownValueError:
            text = ""
    
    text = text.lower()
    
    for keyword, code in WAKE_WORDS.items():
        if keyword in text:
            print(f"\n ☑️  Kiểm tra ----------> ✅ Wake word '{keyword}' detected!\n")
            return code
        
    print("\n ☑️  Kiểm tra ----------> ❌ No wake word detected - Tiếp tục lắng nghe!\n")
    return 0
