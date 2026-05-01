import subprocess
import uuid
import os
import io

# --- 核心路径配置 ---
PIPER_EXE = r"C:\Penn\29AI\piper_windows_amd64\piper\piper.exe"
MODEL_PT = r"C:\Penn\29AI\piper_windows_amd64\models\pt_PT-medium.onnx"
MODEL_ZH = r"C:\Penn\29AI\piper_windows_amd64\models\zh_CN-medium.onnx"
MODEL_XY = r"C:\Penn\29AI\piper_windows_amd64\models\zh_CN-xiao_ya-medium.onnx"
MODEL_CW = r"C:\Penn\29AI\piper_windows_amd64\models\zh_CN-chaowen-medium.onnx"
MODEL_HY = r"C:\Penn\29AI\piper_windows_amd64\models\zh_CN-huayan-medium.onnx"


def generate_piper_audio(text, language="pt", length_scale=None):
    """
    通用 Piper 音频生成函数
    :param text: 要转换的文本
    :param language: 语言标识 ('pt' 或 'zh')
    :param length_scale: 语速调节，不传则使用默认值
    """
    # 1. 根据语言选择模型和默认语速
    if language == "zh":
        model_path = MODEL_ZH
        default_speed = 1.0
    elif language == "hy":
        model_path = MODEL_HY
        default_speed = 1.0
    elif language == "cw":
        model_path = MODEL_CW
        default_speed = 1.0
    elif language == "xy":
        model_path = MODEL_XY
        default_speed = 1.0
    else:
        model_path = MODEL_PT
        default_speed = 1.1

    # 如果调用时传了 length_scale 则覆盖默认值
    speed = length_scale if length_scale is not None else default_speed

    # 2. 基础检查
    if not text:
        return None, "No text provided"
    if not os.path.exists(PIPER_EXE):
        return None, f"找不到 piper.exe: {PIPER_EXE}"
    if not os.path.exists(model_path):
        return None, f"找不到模型文件: {model_path}"

    # 3. 执行合成
    temp_filename = f"temp_tts_{uuid.uuid4().hex}.wav"
    try:
        command = [
            PIPER_EXE,
            "-m", model_path,
            "-f", temp_filename,
            "--length_scale", str(speed)
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=text)

        if process.returncode != 0:
            print(f"Piper Error: {stderr}")
            return None, f"Piper execution failed: {stderr}"

        with open(temp_filename, 'rb') as f:
            return f.read(), None

    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass