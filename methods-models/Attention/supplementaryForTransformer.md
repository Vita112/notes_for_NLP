## 基于[blog：the illustrated transformer](https://jalammar.github.io/illustrated-transformer/)的note
### key notes
> **本篇blog对Transformer的讲解真的十分详细，图文并茂，且逻辑清晰，原文后面附有参考资料，十分棒！！**
> * Encoders和Decoders中 各包含6个子层，每个子层内的结构相同；
> * 使用8个attention headers，对于每一个encoders 和 decoders都有8个随机初始化的矩阵集合，每个集合都被用于将input word embedding(或者来自较低decoder/decoder的向量)投影到不同的representation subspaces，投影过程就是self-attention过程，最后，8个header将得到8个矩阵，将其拼接为一个矩阵(即每个单词的组合表示向量)，然后与 权重矩阵W相乘**得到融合所有注意力头信息的矩阵Z，将其送到FFNN**。
> * 为每个单词的word embedding中加入positional encodinig，以理解输入序列的单词顺序。
> * decoder中，self-attention层被允许处理输入序列中更靠前的那些位置；encoder-decoder attention层工作方式基本与multi-head self-attention层一样，**不同的是：encoder-decoder attention从下面的层创建Q，从堆栈encoder的最终输出中获取K,V**.
### 1 Transformer长什么样？
参考论文笔记

> **Encoder side** 所有encoder在结构上都相同，但没有共享参数，内部有2个子层-self-attention 和 ffnn；

> **decoder side** 所有decoder在结构上也都相同，内部有3个子层- self-attention、encoder-decoder attention和ffnn;  堆栈decoder的输出将被送一个简单的全连接神经网络，经过映射变为一个更大的vector,即对数几率向量logits vector(向量长度为词表大小，每个维度的数字对应某一个单词的分数)。   softmax层将把分数变为概率，获取概率最高的单元格的索引，找到该索引对应的单词，并将它作为这个时间步的输出。
### 2 Encoder side
只有最底层encoder的输入为word embedding + position embedding。

positional encoding有助于确定每个单词的位置，或者序列中不同单词之间的距离，不需要训练，它有产生规则。
#### 2.1 如何计算selt-attention？
* 通过查看输入序列中的其他单词，来获得更好的当前单词表示，是通过上下文理解当前单词的一种办法。*自注意力的另一种解释：编码某个位置单词时，将输入序列中所有单词的表示进行加权求和(因为在计算softmax分数时，某个位置单词的查询向量q会与序列中其他位置的k点乘)，来得到在该位置的输出*。
> step 1: 对于每个单词，创建q，k，v：将带有位置信息的单词嵌入 × 各自的权重矩阵，例：q =  X · W^Q

> step 2: 计算得分，得分决定了对输入序列中的其他单词的关注程度。根据当前单词，对输入序列的每个单词进行评分。

> step 3: divided by 8.   8是论文中的向量k的维数64的平方根，根据经验除以8后，会让梯度更加稳定。

> step 4: softmax。 作用是 使所有单词的分数归一化，得到的分数都为正值，且和为1. softmax分数决定了 每个单词对编码当下位置的贡献。*理论上，在当下位置的单词将获得最高的softmax分数，或者有时关注另一个与当前单词相关的单词也会有帮助*。

> step 5 :  求和加权向量值v. 即用上一步softmax得到的每个word的分数乘以其值向量v，再求和，得到当前位置的自注意力输出。*根据q确定当前位置， 输出包含了来自其他位置的信息*。
#### 2.2 理解multi-headed attention
multi-head的加入，**在2方面提高了自注意力层的性能**：
> 1. 扩展了模型专注于不同位置的能力。在coreference resolution任务中可看到其发挥的作用；

> 2. 可以得到自注意力层的多个representation subspace。Transformer有 8个attention head，因此，对于每个encoder/decoder都有 8个矩阵集合；且对于每一个head，都有独立的Q、K、V矩阵，都分别独立执行相同的self-attention（在8个时间点来计算这些不同的权值矩阵），得到8个不同的z矩阵；将8个z矩阵拼接为一个矩阵，乘以 权值矩阵W^O，*得到multi-head self-attention layer的输出*。

### 3 decoder side
* 分2大块，一个是 6层decoder堆栈起来的decoders，每层decoder的内部结构相同，包含3个子层- self-attention、encoder-decoder attetion和ffnn；一个是 linear + softmax层；

* 解码的每个时间步都会输出 输出序列的一个元素，该元素会在下一个时间步被提供给底端解码器，此外还会拼接表示每个单词位置的position embedding信息，送入decoders中，经过linear transformation 和 softmax处理，得到该时间步的输出。

> **一个时间步 表示这样一个完整的过程：encoder side结束输入序列的编码过程后，得到堆栈encoder的multi-head self-attention输出，该输出被转化为一个包含 键向量k和值向量v的注意力向量集合(K,V)；该注意力向量集合将被送入每一层decoder的encoder-decoder attention sublayer中，经过6层decoder过程得到decoders的输出；然后先对该输出进行一个简单的线性变化得到该位置的score list，softmax将score变为概率，获取概率最高的索引，其对应的单词就是 该时间步的输出**。

* decoders的输出是一个实数向量，① 线性变换层是一个全连接神经网络，它把实数向量投射到 logits向量(对数几率)。*假设训练集词表大小为1万，则对数几率向量将是一个 1万个单元格长度的向量，每个单元格对应都一个单词的分数*。  ②softmax layer把分数变为概率，概率最高的单元格将被选中，它对应的单词则是本次时间步的输出。

### 3 残差The Residuals

注意到不管是在encoder side，还是decoder side，他们各自的sublayer的周围都有一个残差连接，并跟随一个layer-normalization step。

在层归一化步骤中，会对 (word embeddiing X + the output of multi-head self-attention Z)进行归一化。

