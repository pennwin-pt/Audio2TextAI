import numpy as np
from PIL import Image
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard

# 初始化模型（自动下载权重）
model = DalleBart.from_pretrained("kuprel/min-dalle")
processor = DalleBartProcessor.from_pretrained("kuprel/min-dalle")


def generate_image(word: str, output_path: str = "output.png"):
    prompt = f"{word}, cartoon flat style, white background"

    # 处理输入
    inputs = processor([prompt], return_tensors="jax", padding="max_length", truncation=True)
    inputs = shard(inputs.data)

    # 生成图片
    images = model.generate(**inputs, do_sample=True, num_images=4).images
    image = Image.fromarray(np.array(images[0][0]))  # 取第一张

    image.save(output_path)
    print(f"Generated: {output_path}")


# 示例
generate_image("apple", "apple.png")