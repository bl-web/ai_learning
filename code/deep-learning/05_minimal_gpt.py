"""一个可以直接运行的最小 GPT 实验。

它演示了：
1. 字符级 Tokenizer
2. Token Embedding 和位置编码
3. 带因果遮罩的多头自注意力
4. Pre-Norm Transformer Block
5. 下一 token 预测和交叉熵损失
6. 训练后的自回归生成
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 固定随机种子，保证每次运行结果大致一致。
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 字符级 Tokenizer
# ============================================================

text = (
    "今天天气很好。明天天气也很好。今天适合学习。明天继续学习。"
    "机器学习需要多练习。深度学习需要理解公式。模型要学会预测下一个字。"
) * 200

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}


def encode(string):
    return [stoi[char] for char in string]


def decode(token_ids):
    return "".join(itos[token_id] for token_id in token_ids)


data = torch.tensor(encode(text), dtype=torch.long, device=device)

print("================ 数据信息 ================")
print(f"device: {device}")
print(f"文本长度: {len(text)}")
print(f"词表大小 V: {vocab_size}")
print(f"词表: {''.join(chars)}")


# ============================================================
# 2. 模型配置
# ============================================================

block_size = 16       # T：每条训练序列最多有多少个 token
n_embd = 32           # C：每个 token 的向量维度
n_head = 4            # 注意力头数
n_layer = 2           # Transformer Block 数量
batch_size = 16
max_steps = 500
eval_interval = 50
learning_rate = 3e-3

assert n_embd % n_head == 0
head_size = n_embd // n_head


def get_batch():
    """随机抽取一批长度为 block_size 的连续文本片段。"""
    starts = torch.randint(
        low=0,
        high=len(data) - block_size - 1,
        size=(batch_size,),
        device=device,
    )

    x = torch.stack([data[start:start + block_size] for start in starts])
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts])
    return x, y


# ============================================================
# 3. 单头自注意力
# ============================================================

class Head(nn.Module):
    """一个带因果遮罩的单头自注意力。"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        q = self.query(x)   # [B, T, head_size]
        k = self.key(x)     # [B, T, head_size]
        v = self.value(x)   # [B, T, head_size]

        # QK^T / sqrt(d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)
        # scores: [B, T, T]

        # 因果遮罩：只允许当前位置看自己和前面的位置。
        allowed = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=x.device)
        )
        scores = scores.masked_fill(~allowed, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [B, T, T]
        return weights @ v                   # [B, T, head_size]


class MultiHeadAttention(nn.Module):
    """多个单头注意力并行计算并拼接。"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # 每个头输出 [B, T, head_size]，拼接后回到 [B, T, C]。
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)


# ============================================================
# 4. FFN 和 Transformer Block
# ============================================================

class FeedForward(nn.Module):
    """逐 token 独立计算的 FFN：C -> 4C -> C。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-Norm Transformer Block。"""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# 5. 最小 GPT
# ============================================================

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape

        token = self.token_embedding(idx)                    # [B, T, C]
        positions = torch.arange(T, device=idx.device)       # [T]
        pos = self.position_embedding(positions)             # [T, C]
        x = token + pos                                      # [B, T, C]

        x = self.blocks(x)                                   # [B, T, C]
        x = self.ln_f(x)                                     # [B, T, C]
        logits = self.lm_head(x)                             # [B, T, V]

        loss = None
        if targets is not None:
            # get_batch() 已经让 targets[t] = x[t+1]，
            # 所以这里直接把每个位置的 logits 与 targets 对齐即可。
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, start_text, max_new_tokens=14):
        self.eval()

        context = torch.tensor(
            [encode(start_text)],
            dtype=torch.long,
            device=device,
        )  # [1, T]

        for _ in range(max_new_tokens):
            # 这个实验限制总长度不超过 block_size。
            context = context[:, -block_size:]
            logits, _ = self(context)

            # 只使用最后一个位置预测下一个 token。
            next_logits = logits[:, -1, :]                   # [1, V]
            probs = F.softmax(next_logits, dim=-1)           # [1, V]
            next_token = torch.multinomial(probs, num_samples=1)

            context = torch.cat((context, next_token), dim=1)

        self.train()
        return decode(context[0].tolist())


# ============================================================
# 6. 训练
# ============================================================

model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_params = sum(parameter.numel() for parameter in model.parameters())
print("================ 模型信息 ================")
print(f"参数量: {num_params:,}")
print(f"C={n_embd}, T={block_size}, heads={n_head}, layers={n_layer}")

print("================ 开始训练 ================")
for step in range(max_steps + 1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

print("================ 生成结果 ================")
result = model.generate("今天", max_new_tokens=14)
print(result)

