import whisper

model = whisper.load_model("small")
# audio_path = "audio/Ajudar.mp3"  # 替换为你的文件
# audio_path = "audio/Ouvir_150619.wav"  # 替换为你的文件
audio_path = "audio/Ouvir.mp3"

# 转录葡语，输出带时间戳的结果
result = model.transcribe(audio_path, language="pt", verbose=True)
print("转录结果:", result["text"])

# 保存到文件
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
