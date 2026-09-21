import torch
from torch import nn

torch.manual_seed(42)

# 4 个样本，每个样本有 2 个特征
x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

# 0 和 1 表示两个类别
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

# 2 个输入特征，8 个隐藏神经元，2 个输出类别
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(2000):
    # 前向传播，输出形状为 [4, 2]
    logits = model(x)

    # 计算分类损失
    loss = loss_fn(logits, y)

    # 反向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

# 评估模型
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=1)

print("真实标签：", y.tolist())
print("预测标签：", predictions.tolist())
