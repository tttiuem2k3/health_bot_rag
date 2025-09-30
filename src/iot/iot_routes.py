import os
import wave
import subprocess
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import Response
from threading import Thread
import speech_recognition as sr
from gtts import gTTS
from src.wake_detection.test_wake_detection1 import detect_wake_word
from src.rag.rag_chain import get_answer_rag_chain
from src import config
from src.iot.ai_model_api import get_llama3_model
os.makedirs(config.AUDIO_DIRS["chunks"], exist_ok=True)

router = APIRouter()
chunk_counter = 0
processing_status = {
    "is_processing": False,
    "done": False
}
AI = 0
@router.get("/list_music")
async def list_music():
    print("\n----------⭐⭐⭐⭐⭐ Thiết bị phần cứng đã được khởi động!⭐⭐⭐⭐⭐ ----------\n")
    print("\n----------> 🔵 Lấy danh sách bài hát!\n")
    try:
        files = [f for f in os.listdir(config.AUDIO_DIRS["music"]) if f.endswith(".wav")]
        return files
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@router.post("/stream_record")
async def stream_chunk(request: Request):
    global chunk_counter
    chunk_data = await request.body()
    chunk_filename = f"chunk_{chunk_counter:05d}.wav"
    path = os.path.join(config.AUDIO_DIRS["chunks"], chunk_filename)
    with open(path, 'wb') as f:
        f.write(chunk_data)
    chunk_counter += 1
    return {"status": "ok", "file": chunk_filename}

@router.post("/end_stream_record")
async def end_stream():
    combine_chunks(config.AUDIO_DIRS["chunks"], config.AUDIO_DIRS["esp32_record_audio"])
    return {"status": "done", "output": config.AUDIO_DIRS["esp32_record_audio"]}

@router.get("/check_wake_word")
async def check_wake_word():
    global AI
    try:
        has_wake_word = detect_wake_word()
        if has_wake_word == 1:
            AI = 1
            return Response(status_code=200)
        elif has_wake_word == 2:
            AI = 2
            return Response(status_code=202)
        else:
            AI = 1
            return Response(status_code=204)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/start_ai_process")
async def start_ai_process():
    if not processing_status["is_processing"]:
        Thread(target=ai_processing_thread).start()
        return {"message": "Started processing"}
    else:
        return JSONResponse(content={"message": "Already processing"}, status_code=202)

@router.get("/check_status")
async def check_status():
    if processing_status["done"]:
        processing_status["done"] = False
        return {"status": "done"}
    return JSONResponse(content={"status": "false"}, status_code=202)

@router.get("/output_audio")
async def stream_output_audio():
    path = config.AUDIO_DIRS["output_audio_to_esp32"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")

@router.get("/play_music/{filename:path}")
async def stream_music(filename: str):
    if filename.startswith("bao_thuc"):
        folder = config.AUDIO_DIRS["alarm"]
        print("\n----------> 🔔 Phát báo thức!\n")
    elif filename.startswith("VANG"):
        folder = config.AUDIO_DIRS["audio_wake_respon"]
        if AI == 1:
            print("---> 🤖 LISA: vânggg!\n")
        elif AI == 2:
            print("---> 🤖 DAVID: vânggg!\n")
        print("---> 🎙️  Bắt đầu ghi âm câu hỏi của bạn!\n")
    else:
        folder = config.AUDIO_DIRS["music"]
        print(f"\n----------> 🎶 Phát bài hát {filename}!\n")
    safe_path = os.path.join(folder, os.path.basename(filename))
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(safe_path, media_type="audio/wav")


# ---------------- Xử lý AI -----------------

def ai_processing_thread():
    global processing_status
    processing_status["is_processing"] = True
    processing_status["done"] = False
    try:
        recognizer = sr.Recognizer()
        text = ""
        with sr.AudioFile(config.AUDIO_DIRS["esp32_record_audio"]) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="vi-VN")
            except sr.UnknownValueError:
                text = ""
        if AI == 1:
            print(f"---> 🗣  Bạn nói: {text}\n")
            print(f"🧠 LISA đang suy nghĩ câu trả lời......\n")
            response = get_answer_rag_chain(text)
            tts = gTTS(text=response, lang='vi')
            tts.save(config.AUDIO_DIRS["ai_output_audio"])
            print(f"\n---> 🤖 LISA: {response}\n")
        elif AI == 2:
            print(f"---> 🗣  Bạn nói: {text}\n")
            print(f"🧠 DAVID đang suy nghĩ câu trả lời......\n")
            llama_model = get_llama3_model(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            max_completion_tokens=2048,
            temperature=1
            )
            response = llama_model(text)
            tts = gTTS(text=response, lang='vi')
            tts.save(config.AUDIO_DIRS["ai_output_audio"])
            print(f"\n---> 🤖 DAVID: {response}\n")
        convert_to_esp32_wav(config.AUDIO_DIRS["ai_output_audio"], config.AUDIO_DIRS["output_audio_to_esp32"])

    except Exception as e:
        print(f"❌ Lỗi xử lý AI: {e}")

    processing_status["is_processing"] = False
    processing_status["done"] = True

# ----------------- Hàm tiện ích -----------------

def combine_chunks(input_dir, output_file):
    chunk_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])
    with wave.open(output_file, 'wb') as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)  # 16-bit
        wf_out.setframerate(16000)
        for fname in chunk_files:
            with open(os.path.join(input_dir, fname), 'rb') as f:
                wf_out.writeframes(f.read())
    for f in chunk_files:
        os.remove(os.path.join(input_dir, f))
    global chunk_counter
    chunk_counter = 0

def convert_to_esp32_wav(input_path, output_path):
    command = [
        r"D:\App\ffmpeg\ffmpeg-2025-04-17-git-7684243fbe-full_build\bin\ffmpeg.exe",
        "-y",
        "-i", input_path,
        "-filter:a", "volume=1.5, atempo=1.1",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_u8",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"---> 🔊 Phát âm thanh trả lời từ AI!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi convert {input_path}: {e.stderr.decode()}")

"uvicorn iot_routes:router --host 0.0.0.0 --port 8888 --reload --log-level warning --no-access-log"