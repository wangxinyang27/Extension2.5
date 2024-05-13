import torch

# 输入.pth模型文件的路径
model_path = ("checkpoint\\model_epoch_50.pth")
# 加载模型
model = torch.load(model_path)['model']
print(model)

# 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())
print("总参数量:", total_params)
