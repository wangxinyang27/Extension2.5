from matplotlib import pyplot as plt

import scipy.io as sio

# 读取mat文件
mat_file = 'Set5_mat\\baby_GT_x4.mat'
data = sio.loadmat(mat_file)

print(data)

# 提取数据
# data_array = data['data']

# 可视化数据
# plt.plot(data_array)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('MAT Data Visualization')
# plt.show()

import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

dataFile = 'Set5_mat\\baby_GT_x4.mat'
data = scio.loadmat(dataFile)
print(type(data))
# print (data['data'])
# 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
a=data['im_l_ycbcr']
# 取出需要的数据矩阵

# 数据矩阵转图片的函数
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

new_im = MatrixToImage(a)
plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
new_im.save('train_2.png') # 保存图片

