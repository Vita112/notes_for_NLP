## 基于[blog：the illustrated transformer](https://jalammar.github.io/illustrated-transformer/)的note
### key notes
> * Encoders和Decoders中 各包含6个子层，每个子层内的结构相同；
> * 使用8个attention headers，对于每一个encoders 和 decoders都有8个随机初始化的矩阵集合，每个集合都被用于将input word embedding(或者来自较低decoder/decoder的向量)投影到不同的representation subspaces，投影过程就是self-attention过程，最后，8个header将得到8个矩阵，将其拼接为一个矩阵(即每个单词的组合表示向量)，然后与 权重矩阵W相乘**得到融合所有注意力头信息的矩阵Z，将其送到FFNN**。
> * 为每个单词的word embedding中加入positional encodinig，以理解输入序列的单词顺序。
> * decoder中，self-attention层被允许处理输入序列中更靠前的那些位置；encoder-decoder attention层工作方式基本与multi-head self-attention层一样，**不同的是：encoder-decoder attention从下面的层创建Q，从堆栈encoder的最终输出中获取K,V**.
### 1 Transformer长什么样？
参考论文笔记

> **Encoder side** 所有encoder在结构上都相同，但没有共享参数，内部有2个子层-self-attention 和 ffnn；

> **decoder side** 所有decoder在结构上也都相同，内部有3个子层- self-attention、encoder-decoder attention和ffnn;  堆栈decoder的输出将被送一个简单的全连接神经网络，经过映射变为一个更大的vector,即对数几率向量logits vector(向量长度为词表大小，每个维度的数字对应某一个单词的分数)。   softmax层将把分数变为概率，获取概率最高的单元格的索引，找到该索引对应的单词，并将它作为这个时间步的输出。
### 2 Encoder side
只有最底层encoder的输入为word embedding + position embedding。positional encoding不需要训练，它有产生规则。
#### 2.1 如何计算selt-attention？
* 通过查看输入序列中的其他单词，来获得可以更好的当前单词表示，是一种通过上下文理解当前单词的一种办法。
> step 1: 对于每个单词，创建q，k，v：将带有位置信息的单词嵌入 × 各自的权重矩阵，例：q =  X · W^Q

> step 2: 计算得分，得分决定了对输入序列中的其他单词的关注程度。根据当前单词，对输入序列的每个单词进行评分。

> step 3: divided by 8.8是论文中的向量k

> step 4: softmax


