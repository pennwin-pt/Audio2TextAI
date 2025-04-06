from flask import Flask, request, jsonify
from transformers import pipeline
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 加载 Whisper 模型（确保已安装 transformers 和 torch）
model_path = "openai/whisper-small"  # 可替换为本地路径，如 "path/to/your/whisper-model"
try:
    transcriber = pipeline("automatic-speech-recognition", model=model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    transcriber = None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    接收音频文件并返回转录文本
    """
    if not transcriber:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    audio_file = request.files['audio_file']

    # 校验文件格式（仅支持 WAV）
    if not audio_file.filename.endswith('.wav'):
        return jsonify({"status": "error", "message": "Unsupported audio format. Only WAV is supported."}), 415

    try:
        # 临时保存音频文件（可选，避免内存占用过大）
        temp_path = "temp_audio.wav"
        audio_file.save(temp_path)

        # 调用模型进行转录
        result = transcriber(temp_path)
        text = result.get("text", "")

        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({"status": "success", "text": text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # 启动服务
    app.run(host='0.0.0.0', port=5000)
