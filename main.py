import os
from flask import Flask, request, jsonify

import os
import sys

# 自动定位并添加 NVIDIA 库路径，防止 DLL 找不到
try:
    import nvidia.cublas_cu12
    import nvidia.cudnn_cu12
    cublas_path = os.path.dirname(nvidia.cublas_cu12.__file__)
    cudnn_path = os.path.dirname(nvidia.cudnn_cu12.__file__)
    # 将 lib 目录加入系统路径
    os.environ["PATH"] += os.pathsep + os.path.join(cublas_path, "lib")
    os.environ["PATH"] += os.pathsep + os.path.join(cudnn_path, "lib")
except ImportError:
    pass

from faster_whisper import WhisperModel
import torch
from zhconv import convert

app = Flask(__name__)

# --- 核心配置：针对 RTX 4060 (8GB) 优化 ---
# 1. 使用 CTranslate2 版本的 Turbo 模型，速度极快
model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"

print(f"正在加载模型至显卡: {model_size}...")
try:
    # device="cuda" 强制使用显卡
    # compute_type="float16" 是 40系列显卡的最佳运行精度
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("✅ 模型加载成功，显存已占用约 4-5GB")
except Exception as e:
    print(f"❌ 显卡加载失败，请检查 CUDA 环境: {e}")
    # 回退到 CPU (仅作备份，4060 不应该走到这一步)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")


def handle_transcribe_audio(language, custom_prompt):
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    audio_file = request.files['audio_file']
    temp_path = f"temp_{language}.wav"

    try:
        audio_file.save(temp_path)

        # --- 调用 Faster-Whisper 执行转录 ---
        # beam_size=5: 保证识别率
        # vad_filter: 过滤噪音和静音，防止“幻听”
        segments, info = model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            # --- 关键修改点 ---
            initial_prompt=custom_prompt,
            condition_on_previous_text=False,  # 禁止模型根据提示词过度联想
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=200,  # 缩短静音判断，防止短语被切
                speech_pad_ms=300  # 在声音前后多留一点余量
            ),
            # 提高对“无声”的判断阈值，如果真的是杂音，就返回空而不是提示词
            no_speech_threshold=0.6
        )
        # 拼接转录结果
        full_text = "".join([segment.text for segment in segments])

        # 中文繁简转换
        if language == "zh":
            full_text = convert(full_text, 'zh-cn')

        print("AI 理解为", full_text)
        return jsonify({
            "status": "success",
            "text": full_text.strip(),
            "detected_language": info.language,
            "language_probability": info.language_probability
        })

    except Exception as err:
        print("出现异常", str(err))
        return jsonify({"status": "error", "message": str(err)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/transcribe_pt', methods=['POST'])
def transcribe_pt_audio():
    pt_long_prompt = (
        "Este é um áudio em português de Portugal. Contém palavras curtas e frases isoladas. "
        "Por favor, transcreva apenas em português, sem usar inglês."
    )
    return handle_transcribe_audio("pt", pt_long_prompt)


@app.route('/transcribe_zh', methods=['POST'])
def transcribe_zh_audio():
    zh_long_prompt = (
        "这是一段纯正的中文普通话语音，有可能是某个单字、也有可能是一个词组或者是一个短语，但是不太可能是个长句子。 "
        "请务必使用简体中文转录，把这个单字或者词组转录出来，不要夹杂任何英文字母或单词，不要把发音误认为是英文。"
    )
    return handle_transcribe_audio("zh", zh_long_prompt)


if __name__ == '__main__':
    # 确保端口未被占用
    app.run(host='0.0.0.0', port=5000, debug=False)