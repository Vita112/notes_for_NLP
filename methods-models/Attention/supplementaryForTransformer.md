## 基于[blog：the illustrated transformer](https://jalammar.github.io/illustrated-transformer/)的note
### key notes
> * Encoders和Decoders中 各包含6个子层，每个子层内的结构相同；
> * 使用8个attention headers，对于每一个encoders 和 decoders都有8个随机初始化的矩阵集合，每个集合都被用于将input word embedding(或者来自较低decoder/decoder的向量)投影到不同的representation subspaces，投影过程就是self-attention过程，最后，8个header将得到8个矩阵，将其拼接为一个矩阵(即每个单词的组合表示向量)，然后与 权重矩阵W相乘**得到融合所有注意力头信息的矩阵Z，将其送到FFNN**。
> * 为每个单词的word embedding中加入positional encodinig，以理解输入序列的单词顺序。
> * decoder中，self-attention层被允许处理输入序列中更靠前的那些位置；encoder-decoder attention层工作方式基本与multi-head self-attention层一样，**不同的是：encoder-decoder attention从下面的层创建Q，从堆栈encoder的最终输出中获取K,V**.
