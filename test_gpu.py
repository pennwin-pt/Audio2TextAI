import torch
from faster_whisper import WhisperModel


def check_env():
    print("--- 环境检查 ---")
    print(f"1. PyTorch 版本: {torch.__version__}")
    print(f"2. GPU 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"3. 显卡型号: {torch.cuda.get_device_name(0)}")

    print("\n--- 模型加载测试 ---")
    try:
        # 尝试加载 Turbo 模型到 GPU
        # 注意：第一次运行会自动下载模型，可能需要几分钟
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        print("✅ 模型加载成功！你的 4060 已准备好起飞。")
    except Exception as e:
        print(f"❌ 加载失败: {e}")


if __name__ == "__main__":
    check_env()