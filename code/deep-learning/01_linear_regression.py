import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(123)

# 生成数据
x = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * x + 2 + 0.8 * torch.randn_like(x)

# 随机划分训练集和验证集
indices = torch.randperm(len(x))
train_indices = indices[:80]
val_indices = indices[80:]

x_train = x[train_indices]
y_train = y[train_indices]
x_val = x[val_indices]
y_val = y[val_indices]

# 创建线性回归模型
model = nn.Linear(1, 1)

# 损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_losses = []
val_losses = []

# 训练模型
for epoch in range(200):
    # 前向传播
    train_prediction = model(x_train)

    # 计算训练损失
    train_loss = loss_fn(train_prediction, y_train)

    # 反向传播和参数更新
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 验证模型
    with torch.no_grad():
        val_prediction = model(x_val)
        val_loss = loss_fn(val_prediction, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1:3d}, "
            f"train loss: {train_loss.item():.4f}, "
            f"validation loss: {val_loss.item():.4f}"
        )

# 查看模型学到的参数
weight = model.weight.item()
bias = model.bias.item()

print(f"\n学到的权重 w: {weight:.4f}")
print(f"学到的偏置 b: {bias:.4f}")

# 绘制损失曲线
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()