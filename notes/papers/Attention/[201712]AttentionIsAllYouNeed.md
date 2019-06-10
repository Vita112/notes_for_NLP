## Abstract
propose a new simple network architecture：Transformer，完全依赖于attention mechanism。

第一个完全依赖于self-attention来计算(输入和输出)表示的传导模型transduction model,用于处理序列模型相关问题。
## 1 Introduction
RNN，LSTM和GRU等曾被光用于解决 诸如语言模型和机器翻译等 序列建模和传导问题transduction problems。但其有一个缺点：**前后隐藏状态的依赖性导致无法进行并行计算**。传统的方法大多采用RNN 和 CNN作为encoder-decoder的模型基础，Transformer模型没有用任何CNN 或者RNN结构，并实现了并行运算。
## 2 Background
在使用CNN作为模型构建基块的传统方法，如Extended Neutal GPU，ByteNet，ConcS2S等，将两个任意输入或输出位置的信号联系起来所需要的操作次数 随着这两个位置间距离的增加而增加。**因而，远距离学习更加困难。在Transformer中，这个操作次数被缩减为一个常数，尽管由于平均 加权注意力位置降低了有效分辨率，我们使用Multi-Head Attention来抵消这种影响**。

self-attention，又称为intra-attention，是一种 联系single sequence的不同位置来计算序列表示的注意力机制。已经在阅读理解，抽象摘要，文本蕴含等任务上取得了成功。
## 3 model architecture
大多数自然语言转换模型都包含一个encoder-decoder结构，模型输入是一个离散符号序列x=（x1，x2，……，xn）。encoder负责将输入序列映射为 连续值序列z=(z1,z2,……,zn)，给定z，decoder以一次生成一个元素的方式生成符号的输出序列y=(y1,y2,……,yn).下图是Transformer的模型构造：
![transformer_model_architecture]()
### 3.1 堆栈encoder和decoder
> Encoder

由6个相同层的堆栈构成，每层中包含2个子层sub-layers，分别是multi-head self-attention mechanism 和 simple，position-wise fully connected feed-forward network；对每一层进行正则化后，在2个子层的每一层周围残差连接residual connection（**为什么使用残差连接？达到了怎样的效果？**）。为方便residual connection，模型中的所有子层，包括embedding layers，其输出的维数都为d_model=512.
> Decoder

同样，由6个相同层的堆栈构成，除encoder中的2个子层外，还插入了第3个子层**（为什么增加一层，作用何在？）**，这个子层在encoder stack的输出上执行multi-head attention。同样，在每一层进行正则化后，在每一个子层周围使用residual connection。同样修正了self-attention sub-layer来防止模型关注后续位置的信息，保证位置i的预测仅依赖于前i-1个位置的已知输出。
### 3.2 attention
注意力机制可被看做 将一个查找query和一个键值对key-value pairs集合映射为一个输出的过程，其中，query，keys，values以及output都是vectors。输出是一个values的加权和，此处分配给每个value的weight通过对应key的query的兼容函数得到。本文的注意力机制细节如下图：
![multi-head_attention]()

> scaled dot-product attention

**input**:d_k维的queries和keys，d_v维的values。**then**:计算所有keys和queries的点积，除以$\sqrt{d_{k}}$，通过softmax函数得到values的权重。在实际操作中，由于同时在一个queries的集合上计算注意力，我们将queries，keys和values都打包进matrix，即得到Q,K,V。公式表示如下：
$$Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$$

为了解决d_k很大时，dot-product的结果将变得很大，导致softmaxd在求导时遇到梯度消失的问题，使用$\sqrt{d_{k}}$进行归一化处理。
+ 此处提到的additive attention和dot-product attention，具体指的是什么？
> multi-head attention

将d_model 维的queries，keys和values分别线性映射h次成 不同的、学习到的$d_k$维,$d_k$维和$d_v$维。在h次映射后得到的结果上并行使用注意力，得到一个$d_v$维输出。
## 4 self-attention
## 5 training
## 6 results
## 7 conclusion

> references

1. [softmax函数及其导数](https://blog.csdn.net/cassiePython/article/details/80089760)

2. [《Attention Is All You Need 》阅读笔记](https://blog.csdn.net/songbinxu/article/details/80332992)
