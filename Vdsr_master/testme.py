import torch
from PIL import Image

# 定义函数来读取图像并转换为张量
def image_to_tensor(image_path):
    # 使用PIL库打开图像
    image = Image.open(image_path)

    # 将图像转换为张量
    transform = torch.nn.functional.to_tensor
    tensor_image = transform(image)

    return tensor_image


# 图像文件路径
lr_path = "benchmark/Set5/LR_bicubic/X4/baby_x4.png"
# 将图像转换为张量
lr_tensor = image_to_tensor(lr_path)
# 打印张量的形状
# print("Image tensor shape:", image_tensor.shape)

# 模型
model_path = ""
