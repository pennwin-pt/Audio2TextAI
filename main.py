import datetime

import whisper
from flask import Flask, request, jsonify
import torch
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 检查 CUDA 是否可用，并选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model = whisper.load_model("medium").to(device)
    # model = whisper.load_model("small").to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

@app.route("/shutdown", methods=["GET"])
def shutdown():
    """
    验证参数，是个日期格式20250506这种格式，如果日期正确，就执行关机任务
    :return:
    """
    # 获取日期参数
    date_str = request.args.get('date')

    if not date_str:
        return jsonify({"status": "error", "message": "缺少日期参数"}), 400

    # 验证日期格式
    try:
        # 将字符串转换为日期对象
        input_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        return jsonify({"status": "error", "message": "日期格式不正确，应为YYYYMMDD格式"}), 400

    # 获取当前日期
    current_date = datetime.datetime.now().date()

    if input_date == current_date:
        try:
            # 执行关机命令（根据操作系统不同）
            if os.name == 'nt':  # Windows 系统
                os.system("shutdown /s /t 1")
            else:  # Linux/Mac 系统
                os.system("shutdown -h now")
            return jsonify({"status": "success", "message": "关机命令已执行"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"执行关机命令失败: {str(e)}"}), 500
    else:
        return jsonify({"status": "error", "message": "日期不正确"}), 400

@app.route('/transcribe_pt', methods=['POST'])
def transcribe_pt_audio():
    """
    接收音频文件并返回转录文本
    """
    return handle_transcribe_audio("pt")

@app.route('/transcribe_zh', methods=['POST'])
def transcribe_zh_audio():
    """
    接收音频文件并返回转录文本
    """
    return handle_transcribe_audio("zh")

@app.route('/transcribe_en', methods=['POST'])
def transcribe_en_audio():
    """
    接收音频文件并返回转录文本
    """
    return handle_transcribe_audio("en")


def handle_transcribe_audio(language):
    if not model:
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
        result = model.transcribe(temp_path, language=language, verbose=True)
        text = result["text"]
        if language == "zh":
            from zhconv import convert
            text = convert(text, 'zh-cn')  # 繁体转简体

        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({"status": "success", "text": text})
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 500


if __name__ == '__main__':
    # 启动服务
    app.run(host='0.0.0.0', port=5000)
