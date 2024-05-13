import os
import h5py
import numpy as np
from PIL import Image

# 输入目录路径
input_directory = "dataset"  # 替换为你的输入目录路径

# 创建 H5 文件
output_h5_path = "train.h5"  # 输出 H5 文件路径
with h5py.File(output_h5_path, "w") as h5_file:
    # 创建 data 数据集组
    data_group = h5_file.create_group("data")

    # 创建 label 数据集组
    label_group = h5_file.create_group("label")

    data_list = []
    label_list = []

    # 遍历输入目录中的 data 和 label 子目录

    label_path = os.path.join(input_directory, "label")
    for image_path in os.listdir(label_path):
        _image = Image.open(label_path + "\\" + image_path).convert("L")
        image_array = np.array(_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        label_list.append(image_array)
    label_numpy = np.stack(label_list)

    data_path = os.path.join(input_directory, "data")
    for image_path in os.listdir(data_path):
        _image = Image.open(data_path + "\\" + image_path).convert("L")
        image_array = np.array(_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        data_list.append(image_array)
    date_numpy = np.stack(data_list)

    print(date_numpy.shape, label_numpy.shape)

    data_group.create_dataset("data", data=date_numpy)
    label_group.create_dataset("label", data=label_numpy)


print("H5 文件生成完成。")
