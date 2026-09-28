import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)

# 三个 token，用 4 维向量表示。这里只看计算过程，不讨论向量含义。
tokens = ["我", "喜欢", "猫"]
x = torch.tensor([
    [0.2, 0.1, 0.7, 0.4],
    [0.9, 0.3, 0.1, 0.6],
    [0.4, 0.8, 0.5, 0.2],
])

num_tokens, d_model = x.shape
d_k = 2
scale = d_k ** 0.5

# 三个线性层分别把输入转换成 Q、K、V。
w_q = nn.Linear(d_model, d_k, bias=False)
w_k = nn.Linear(d_model, d_k, bias=False)
w_v = nn.Linear(d_model, d_k, bias=False)

q = w_q(x)
k = w_k(x)
v = w_v(x)

# 标准自注意力：先用 Q 和 K 计算相似度，再除以 sqrt(d_k) 防止数值过大。
scores = q @ k.T
scaled_scores = scores / scale
attention_weights = torch.softmax(scaled_scores, dim=-1)

# 用注意力权重对 V 做加权求和，得到每个 token 的新表示。
output = attention_weights @ v

# 用循环重新算一遍，帮助确认矩阵乘法的每一步在做什么。
manual_weights = []
for i in range(num_tokens):
    one_query_scores = []
    for j in range(num_tokens):
        one_query_scores.append(torch.dot(q[i], k[j]) / scale)

    one_query_scores = torch.stack(one_query_scores)
    manual_weights.append(torch.softmax(one_query_scores, dim=0))

manual_weights = torch.stack(manual_weights)
assert torch.allclose(manual_weights, attention_weights)

print("张量形状")
print(f"X:       {tuple(x.shape)}  => [token数量, 输入维度]")
print(f"Q/K/V:   {tuple(q.shape)}  => [token数量, 查询或键的维度]")
print(f"scores:  {tuple(scores.shape)}  => [每个token, 所有token]")
print(f"output:  {tuple(output.shape)}  => [token数量, 输出维度]")
print(f"每行权重之和: {attention_weights.sum(dim=-1).tolist()}")

print("\n注意力权重：行是当前 token，列是它关注的 token")
print("            " + "".join(f"{token:>10}" for token in tokens))
for token, row in zip(tokens, attention_weights):
    values = "".join(f"{value:10.3f}" for value in row)
    print(f"{token:>4}  ->  {values}")

print("\n每个 token 的新表示")
for token, vector in zip(tokens, output):
    print(f"{token}: {vector.tolist()}")

print("\n练习：")
print("1. 把 d_k 改成 4，观察张量形状和输出维度。")
print("2. 暂时删掉除以 scale 的步骤，观察注意力权重是否更集中在少数 token。")
print("3. 修改 x 的第一行，观察 Q/K/V、权重和 output 中哪些行发生变化。")
