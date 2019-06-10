## Abstract
propose a new simple network architecture：Transformer，完全依赖于attention mechanism。第一个完全依赖于self-attention来计算(输入和输出)表示的传导模型transduction model,用于处理序列模型相关问题。
> **创新点**

提出自注意力self-attention，每个词和所有词计算attention，使每个词都有全局的语义信息(长依赖性)。*不管两个词之间的距离有多长，最大的路径长度也都只是1*。

提出multi-head atention，不同的head学习不同的子空间语义。
## 1 Introduction
RNN，LSTM和GRU等曾被光用于解决 诸如语言模型和机器翻译等 序列建模和传导问题transduction problems。但其有一个缺点：**前后隐藏状态的依赖性导致无法进行并行计算**。传统的方法大多采用RNN 和 CNN作为encoder-decoder的模型基础，Transformer模型没有用任何CNN 或者RNN结构，并实现了并行运算。
## 2 Background
在使用CNN作为模型构建基块的传统方法，如Extended Neutal GPU，ByteNet，ConcS2S等，将两个任意输入或输出位置的信号联系起来所需要的操作次数 随着这两个位置间距离的增加而增加。**因而，远距离学习更加困难。在Transformer中，这个操作次数被缩减为一个常数，尽管由于平均 加权注意力位置降低了有效分辨率，我们使用Multi-Head Attention来抵消这种影响**。

self-attention，又称为intra-attention，是一种 联系single sequence的不同位置来计算序列表示的注意力机制。已经在阅读理解，抽象摘要，文本蕴含等任务上取得了成功。
## 3 model architecture
大多数自然语言转换模型都包含一个encoder-decoder结构，模型输入是一个离散符号序列x=（x1，x2，……，xn）。encoder负责将输入序列映射为 连续值序列z=(z1,z2,……,zn)，给定z，decoder以一次生成一个元素的方式生成符号的输出序列y=(y1,y2,……,yn).下图是Transformer的模型构造：

![transformer_model_architecture](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/transformer_model_architecture.jpg)
### 3.1 堆栈encoder和decoder
![encoder-decoder结构](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/encoder-decoder%E7%BB%93%E6%9E%84.png)
> **Encoder**

![encoder](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/encoder.png)

由6个相同层的堆栈构成，每层中包含2个子层sub-layers，分别是multi-head self-attention mechanism 和 simple，position-wise fully connected feed-forward network；对每一层进行正则化后，在2个子层的每一层周围残差连接residual connection（**为什么使用残差连接？达到了怎样的效果？**）。为方便residual connection，模型中的所有子层，包括embedding layers，其输出的维数都为d_model=512.
> **Decoder**

![decoder](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/decoder.png)

同样，由6个相同层的堆栈构成，除encoder中的2个子层外，还插入了另一个对ENCODER的attention子层**（为什么增加一层，作用何在？）**，这个子层在encoder stack的输出上执行multi-head attention。同样，在每一层进行正则化后，在每一个子层周围使用residual connection。同样修正了self-attention sub-layer来防止decoder关注后续位置的信息，保证位置i的预测仅依赖于前i-1个位置的已知输出(具体地，添加一个mask将位置i及其之后的token遮盖住)。
### 3.2 attention
**注意力机制可被看做 将一个查找query和一个键值对key-value pairs集合映射为一个输出的过程**，其中，query，keys，values以及output都是vectors。输出是一个values的加权和，此处分配给每个value的weight通过对应key的query的兼容函数得到。本文的注意力机制细节如下图：

![scaled_dot-product_attention_and_multi-head_attention]()
> **scaled dot-product attention**

+ 序列问题中传统的attention:

![traditional_attention_in_sequence_modeling](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/traditional_attention_in_sequence_modeling.png)

+ k, v, q表示方式下的scaled dot-product attention

**input**:d_k维的queries和keys，d_v维的values。**then**:计算所有keys和queries的点积，除以$\sqrt{d_{k}}$，通过softmax函数得到values的权重。在实际操作中，由于同时在一个queries的集合上计算注意力，我们将queries，keys和values都打包进matrix，即得到Q,K,V。公式表示如下：
$$Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$$
其中，矩阵Q,K,V形状分别如下：

![matrix_Q_K_V](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/matrix_Q_K_V.png)

如果没有激活函数，运算后得到的是一个 n×dv维的矩阵(n是句子长度),即attention层将原始序列n×dk 编码成了一个新的序列n×dv。上述公式的向量表示为：
$$Attention(q_{t},K,V)=\sum_{s=1}^{m}\frac{1}{Z}exp(\frac{<q_{t},k_{s}>}{\sqrt{d_{k}}})v_{s}$$
上述公式可解释为
> qt是一个查询query，通过qt与ks的点积，经过softmax函数，得到qt与vs的相似度，后加权求和，得到一个dv维的向量。

这其实是传统attention的一个变种：

![ATT_from_tradition_to_Scaled_dot-product_att](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/ATT_from_tradition_to_Scaled_dot-product_att.png)

为了解决d_k很大时，dot-product的结果将变得很大，导致softmaxd在求导时遇到梯度消失的问题，使用$\sqrt{d_{k}}$进行归一化处理。

+ **此处提到的additive attention和dot-product attention，具体指的是什么**？
> additive attention加性注意力：使用一个 有隐藏层的全连接前馈网络 来计算注意力分配

![additive_attention](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/additive_attention.png)
> dot-product attention点乘注意力

![dot-product attention](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/dot-product_attention.png)

> **multi-head attention**

将d_model 维的queries，keys和values分别线性映射h次成 不同的、学习到的$d_k$维,$d_k$维和$d_v$维。在h次的每一次映射后得到的结果(queries,keys,values)上并行执行注意力（即进行scaled dot-product attention operation），返回一个$d_v$维输出。将所有输出拼接，再进行一次线性映射，得到最终的结果值。multi-head attention是由若干个并行运行的attention layers组成，**允许模型联合关注来自不同位置的不同表示子空间的信息**。具体地，在encoder-decoder框架中，query来自上一层decoder，而key和value则是上一层encoder的输出。正是这种机制，使得句子中每一个part都可以参与到encoder-decoder的过程。

### 3.3 decoder
![computation_flow_of_decoder1](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computation_flow_of_decoder.jpg)
![computation_flow_of_decoder2](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computation_flow_of_decoder2.jpg)
细节动态图[click](https://www.zhihu.com/question/61077555/answer/183884003)

## 3.4 一个例子——input：2个单词——thinking 和 machines；计算得到q,k,v,得到self-attention的输出z

![computations_of_self-attention1](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computations_of_self-attention1.jpg)
![computations_of_self-attention2](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computations_of_self-attention2.jpg)
![multi-head_self-attention1](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_self-attention.jpg)
![multi-head_self-attention2](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_self-attention2.jpg)
![multi-head_self-attention3](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_self-attention3.jpg)
## 4 self-attention
![the_whole_computation_flow_of_self-attenion](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/the_whole_computation_flow_of_self-attenion.jpg)
## 5 training
## 6 results
## 7 conclusion

> references

1. [softmax函数及其导数](https://blog.csdn.net/cassiePython/article/details/80089760)

2. [《Attention Is All You Need 》阅读笔记](https://blog.csdn.net/songbinxu/article/details/80332992)

3. [attention_is_all_you_need解读](https://zhuanlan.zhihu.com/p/34781297)

4. [如何理解谷歌团队的机器翻译新作《Attention is all you need》？](https://www.zhihu.com/question/61077555/answer/183884003)
