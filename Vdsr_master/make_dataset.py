import os
from PIL import Image
import random

# 输入目录路径
LR_directory = "DIV2K/DIV2K_train_LR_bicubic/X4"
HR_directory = "DIV2K/DIV2K_train_HR"

# 输出目录路径
output_directory = "your_output_directory"  # 替换为你的输出目录路径

# # 创建输出目录
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# 读取低分辨率图像并对其进行 Bicubic 插值
for filename in os.listdir(LR_directory):
    if filename.endswith(".png"):
        # 读取低分辨率图像
        LR_image = Image.open(LR_directory + "\\" + filename)
        HR_image = Image.open(HR_directory + "\\" + filename.replace("x4", ""))

        _width, _height = HR_image.size

        # 对低分辨率图像进行 Bicubic 插值，扩大为原来的 4 倍
        LR_Bicubic_image = LR_image.resize((_width, _height), Image.BICUBIC)
        # 随机选择裁剪区域的左上角坐标
        left = random.randint(0, _width - 41)
        top = random.randint(0, _height - 41)

        # 裁剪图像上的 41x41 区域
        LR_Bicubic_cropped_image = LR_Bicubic_image.crop((left, top, left + 41, top + 41))
        HR_cropped_image = HR_image.crop((left, top, left + 41, top + 41))

        # 构造输出图像文件路径
        LR_Bicubic_save_path = "dataset/data"
        HR_save_path = "dataset/label"

        filename = str(int("1" + filename.replace("x4.png","")) + 900)[1:]
        filename = filename + "x4.png"

        # 保存裁剪后的图像
        LR_Bicubic_cropped_image.save(LR_Bicubic_save_path + "\\" + filename)
        HR_cropped_image.save(HR_save_path + "\\" + filename.replace("x4", ""))

    print(filename.replace("x4.png", ""))
    if filename.replace("x4.png", "") == "1000":
        break
print("所有图像处理完成并保存。")
