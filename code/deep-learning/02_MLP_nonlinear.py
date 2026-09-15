import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# 生成非线性数据：y = x² + 噪声
x = torch.linspace(-2, 2, 200).reshape(-1, 1)
y = x ** 2 + 0.2 * torch.randn_like(x)

# 随机划分训练集和验证集
indices = torch.randperm(len(x))
train_indices = indices[:160]
val_indices = indices[160:]

x_train = x[train_indices]
y_train = y[train_indices]
x_val = x[val_indices]
y_val = y[val_indices]

# 线性模型
linear_model = nn.Linear(1, 1)

# 多层感知机
mlp_model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.MSELoss()

linear_optimizer = torch.optim.SGD(
    linear_model.parameters(),
    lr=0.01
)

mlp_optimizer = torch.optim.SGD(
    mlp_model.parameters(),
    lr=0.01
)

linear_losses = []
mlp_losses = []

for epoch in range(1000):
    # ===== 训练线性模型 =====
    linear_prediction = linear_model(x_train)
    linear_loss = loss_fn(linear_prediction, y_train)

    linear_optimizer.zero_grad()
    linear_loss.backward()
    linear_optimizer.step()

    # ===== 训练多层感知机 =====
    mlp_prediction = mlp_model(x_train)
    mlp_loss = loss_fn(mlp_prediction, y_train)

    mlp_optimizer.zero_grad()
    mlp_loss.backward()
    mlp_optimizer.step()

    linear_losses.append(linear_loss.item())
    mlp_losses.append(mlp_loss.item())

    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch {epoch + 1:4d}, "
            f"linear loss: {linear_loss.item():.4f}, "
            f"MLP loss: {mlp_loss.item():.4f}"
        )

# 按 x 的大小排序，方便画出平滑曲线
sort_indices = torch.argsort(x_val[:, 0])
x_plot = x_val[sort_indices]

with torch.no_grad():
    linear_prediction = linear_model(x_plot)
    mlp_prediction = mlp_model(x_plot)

# 绘制预测结果
plt.scatter(x_train, y_train, s=10, label="data")
plt.plot(x_plot, linear_prediction, label="linear model")
plt.plot(x_plot, mlp_prediction, label="MLP")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 绘制损失曲线
plt.plot(linear_losses, label="linear loss")
plt.plot(mlp_losses, label="MLP loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()