## 单头注意力与多头注意力
- 前置设定：  
B: batch size，每次处理多少个句子  
T：sequence length,一个句子有多少个token  
C: d_model,一个token的向量维度  
h: 注意力头数  
d_head: 每个注意力头里面一个token的向量维度
- 单头注意力：每个token都只有一套Q,K,V，一般shape为` [B,T,d_k] ` ` [B,T,d_k] ` `[B,T,d_v] `

$$
\mathrm{new_x}=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- 为什么要除以 $\sqrt{d_k}$ 呢？
- 因为softmax对于差距过大的情况不能很好地处理，会进入饱和区间，梯度过小，模型很难学习。所以我们要除以 $\sqrt{d_k}$ 来进行放缩，把差距控制的小一些，使模型能够平滑调整 
- 多头注意力：一个头只能学到一套注意力机制，而多个头则可以形成多套注意力分布，表达能力更强，但至于每个头学到了什么，我们也不得而知。
- 多头注意力得到新的词向量的过程：  
让` [B,T,C] `经过不同的线性投影，得到h组Q,K,V,其中，他们应该满足 $h \times d_{\mathrm{head}}=C$ ，然后不同组别的Q,K,V分别进行与单头注意力机制相同的计算，再进行拼接，最后再经过W_O(线性投影)，将不同的信息融合在一起得到新的词向量。

## 位置编码
- 一个句子中token的位置也很重要，如：“猫追狗” “狗追猫”，位置也蕴含着信息
- 通常我们会生成一个位置信息的表，然后将查表得到的词向量加上位置向量
-  $x=\mathrm{TokenEmbedding}+\mathrm{PositionEmbedding}$ 
- 位置向量表生成过程：  
随机生成一个[l_max,C]的矩阵，l_max是提前指定好的最大的位置数，输入[B,T,C]=[2,5,524],就取前5行加到输入里面，然后经过学习的过程更新参数。
- 现代llm也使用RoPE，通过旋转Q,K来得到相对位置之间的关系

## 残差连接
- 浅层的梯度必须穿过后面的层，如果深层有多层梯度小于1，就会出现浅层得到的梯度非常小，几乎收不到有效的梯度，难以学习。
- 残差连接： $x=x+F(x)$ 
- 残差连接过程：  
1. 整个输入的流程：` x→A层→y=A(x)→B层→z=B(y)+y `
2. 梯度计算的过程：L为损失值，B层的梯度为：

$$
\frac{\partial L}{\partial w_B}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial w_B}
$$

A层的梯度为:

$$
\frac{\partial L}{\partial w_A}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial y}\frac{\partial y}{\partial w_A}=\left(1+\frac{\partial B(y)}{\partial y}\right)\frac{\partial L}{\partial z}\frac{\partial y}{\partial w_A}
$$

也就是说给梯度提供了一个可以不经过B层就往后传递的一个路径。
- 可能原来一层神经网络收到的梯度本应没有这么大，但残差连接让其变大，这有影响吗？
- 残差连接主要是为了让深层的神经网络更好的学习，不至于因为梯度消失而没办法学习，让梯度变大会有影响，但好过没办法学习。

## Layernorm

$$
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
$$

-  $\gamma$ , $\beta$  都是shape=[C]的向量，他们也是学习而来
- 通过LayerNorm，每个token的C维向量单独归一化，均值约为0，方差约为1
- 为什么要这样做呢？  
目的是稳定每个 token 的特征分布，避免激活值和注意力分数尺度失控，使深层网络更容易训练。但同时他也会损失一些绝对尺度的信息

## FFN
- 又叫前馈神经网络，将一个token对应的向量先升维再降维，使他获得表达非线性的能力
- 具体过程：` linear(C,d_ff)→ReLU/GELU→linear(d_ff,C) `
- 注意：是每个token独立处理，token之间不影响，输入输出shape均为[B,T,C]

## 因果遮罩
- 在模型训练的时候，模型只能看到自己对应的token和他前面的token，不能让它看到后面的token，否则会提前泄露答案。
- 因果遮罩实现过程：  
1. 在得到每个位置都有对应分数的矩阵之后，将其变成一个下三角矩阵
2. 主对角线上面的所有分数都改为-inf ,即负无穷
3. 通过softmax获得每个位置的概率，其中-inf对应位置概率为0
- 因果遮罩用于 $\frac{QK^\top}{\sqrt{d_k}}$ 之后，以得到每个位置对应的权重

## Transformer block
- 一个完整的block的公式如下：

$$
x_1=x+\mathrm{MultiHeadAttetion}(\mathrm{LayerNorm}(x))
$$

$$
x_2=x_1+\mathrm{FFN}(\mathrm{LayerNorm}(x_1))
$$

- 一个完整的Transformer的流程，通常要经过很多个block，且每个block输出的shape都是[B,T,C]

## Pretraining与生成答案
- 预训练：给模型喂大量的语料，通过自监督学习的过程，建立起base-model
- Pretraining可以并行计算，同时计算多个token的预测结果，并计算损失求平均统一更新参数  
但是生成答案的时候只能一个token一个token的生成，并反复将其添加到输入后面继续进行下一个token的预测
- 选择token的策略：  
1. Greedy：直接选择概率最高的token，即` next_token=logits.argmax(dim=-1) `
2. Temperature:控制各个token的平滑程度

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

- τ>1时，分布更平滑、随机，token之间的差距更小  
τ=1时，不改变分布的差距  
τ<1时，分布更尖锐，保守，但生成更单一
3. Top-k:只留前k个，重新归一化之后采样选择  
4. Top-p:按照概率由高到低排序，只留概率相加到p的几个token，重新归一化之后采样选择

## SFT与偏好对齐
- SFT：让模型能够按照指令和问题进行回答，对话，训练的时候将数据拆分为问题与回答，重点训练模型怎么做出更好的回答。
- 偏好对齐：让回答更符合人类的偏好，包括RLHF和DPO
- RLHF：
$$
\max_\theta\mathbb{E}\left[r_\phi(x,y)-\beta D_{\mathrm{KL}}(\pi_\theta\parallel\pi_{\mathrm{ref}})\right)
$$

## KV cache
- 我们在生成答案的时候，前面几个token的K,V由于后面继续预测的时候还需要用到，所以会将其存起来，这样可以不用重复计算

## LoRA
- 模型在针对某一类任务进行训练的时候，有时候并不需要全量微调，于是冻结W0,引入LoRA。
- 我们指定将LoRA运用到哪几块，用到W_q,W_k等等，都由我们决定
- 前向传播的公式：$W=W_0x+BAx$
- 其中，W_0=[n,n]不更新，我们只更新B,A,B_shape=[n,r],A_shape=[r,n]，r为超参，A一开始随机化，B一开始全部赋值为0，之后通过学习更新参数，更新参数也是LoRA真正节省算力的一步。