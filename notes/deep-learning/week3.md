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