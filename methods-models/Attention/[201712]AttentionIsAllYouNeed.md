## Abstract
propose a new simple network architecture：Transformer，完全依赖于attention mechanism。第一个完全依赖于self-attention来计算(输入和输出)表示的传导模型transduction model,用于处理序列模型相关问题。

code available：https://github.com/tensorflow/tensor2tensor

paper link [here](https://arxiv.org/pdf/1706.03762.pdf)
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

同样，由6个相同层的堆栈构成，除encoder中的2个子层外，还插入了另一个对ENCODER的attention子层**（为什么增加一层，作用何在？）**，这个子层在encoder stack的输出上执行multi-head attention。同样，在每一层进行正则化后，在每一个子层周围使用residual connection。**修正了self-attention sub-layer来防止decoder关注后续位置的信息，保证位置i的预测仅依赖于前i-1个位置的已知输出(具体地，添加一个mask将位置i及其之后的token遮盖住)**。

![masked_self-attention_with_softmax](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/masked_self-attention_with_softmax.jpg)
### 3.2 attention
**注意力机制可被看做 将一个查找query和一个键值对key-value pairs集合映射为一个输出的过程**，其中，query，keys，values以及output都是vectors。输出是一个values的加权和，此处分配给每个value的weight通过对应key的query的兼容函数得到。本文的注意力机制细节如下图：

![scaled_dot-product_attention_and_multi-head_attention](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_attention.jpg)
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

将d_model 维的queries，keys和values分别线性映射h次成 不同的、学习到的$d_k$维,$d_k$维和$d_v$维。在h次的每一次映射后得到的结果(queries,keys,values)上并行执行注意力（即进行scaled dot-product attention operation），返回一个$d_v$维输出。将所有输出拼接，再进行一次线性映射，得到最终的结果值。multi-head attention是由若干个并行运行的attention layers组成，**允许模型联合关注来自不同位置的不同表示子空间的信息，这里主要重点在不同子空间，因为存在多个head**。

> Transformer use multi-head in 3 ways:

① 在encoder-decoder attention layers中，query来自previous decoder layer，key和value来自encoder output，这使得decoder中的每个位置都可以处理输入序列中的所有位置。

② encoder中包含多个self-attention layers，每一个self-attention layer 的所有keys，values和queries 均来自 the output of ther previous layer in the encoder.**encoder中的每一个位置 可以关注该encoder的previous layer 的所有位置**。

③ decoder中也包含多个self-attention layers，这些layers 允许decoder中的每个位置关注 该位置之前的所有位置的信息(包括该位置)。

**此处有两个问题：1，多头的head个数 如何确定？2.位置信息直观上看，具体指什么呢？**

### 3.3 self-attention
在解释为什么使用self-attention时，从以下3各方面考虑：
> 1. the computational complexity per layer;
> 2. the amount of computation that can be parallelized;
> 3. the path length between long-range dependencies in the network.影响学习长期依赖能力的一个关键因素是：网络中需要遍历的前向和后向信号的路径长度。

下图显示了不同层类型的complexity per layer，sequential operarions以及maximun path length：

![comparison_of_different_layer_types]()

由上图可知，一个自注意力层只需要 一个常数级的序列操作 将所有位置连接起来，而一个递归层则需要O(n)个序列操作。**计算复杂度**上，当n小于d时，自注意力比递归层快，而在机器翻译的先进模型中，序列长度n都比表示维度d小，例如字段表示和字节表示(word-piece and byte-pair representations).

一个单独的核窗口宽度为 k<n 的卷积层并不会连接所有的输入和输出位置对。而要连接所有的位置对，使用相邻核(contiguous kernels)时，需要堆栈O(n/k)的卷积层；使用膨胀卷积(dilated convolution)时，需堆栈O(logk(n))的卷积层。

**关于不同类型的卷积操作，可参考知乎的两篇文章**：
> 1. [一文了解各种卷积结构原理](https://zhuanlan.zhihu.com/p/28186857)
> 2. [CNN中千奇百怪的卷积方式](https://zhuanlan.zhihu.com/p/29367273)

下图是一个自注意力的完整计算流程：

![the_whole_computation_flow_of_self-attenion](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/the_whole_computation_flow_of_self-attenion.jpg)

**使用self-attention的另一个好处是：可以返回更易解释的模型**。不仅单个的注意力头可以明确地学习执行不同的任务，有许多展现出了与句子的语法和语义结构相关的行为。

### 3.4 position-wise feed-forward networks
在encoder和decoder中，均使用了定位全链接前馈网络，**它应用于每个位置，并且完全相同**。公式如下：
$$FFN(x)=max(0,xW_{1}+b_{1})W_{2}+b_{2}$$
同一层上跨不同位置的线性变化是一样的，但是使用不同的参数from layer to layer。
### 3.5 others
+ embeddings and softmax

使用学习到的embeddings将input tokens和output tokens 转化为d_model维的向量；使用学习到的线性变换和softmax函数转换decoder output，用于预测下一个token的概率。**in this model，我们在两个embedding layers 和 pre-softmax linear transformation之间，使用相同的权重矩阵**。
+ positional encoding

在encoder和decoder stacks的底层，将positional encodings添加到input embeddings 中，以利用序列的顺序信息。**in this work，使用不同频率的sine和cosine函数，公式如下**：
$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})\\\\
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$$
此处，pos代表position，i代表dimension，也就是说:**positional encoding的每一个dimension都对应一个正弦曲线sinusoid。波长形成了一个从2π到10000·2π的geometric progression几何数列
## 4 一个例子-multi-head self-attention model

+ input：2个单词——thinking 和 machines；

+ 首先embedding得到单词表示wj，矩阵点乘计算得到input每个单词表示wj 的三个向量 qi,kj,vj；*此处，i，j分别表示input sequence和output sequence的索引，且位置对应*。

![computations_of_self-attention1](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computations_of_self-attention1.jpg)

+ 对于output sequence中的yi，计算向量qi，kj点乘，得到相似性score，score规范化后进行softmax，得到score的概率分布，即是预测yi时，给与input sequence中每个单词的注意力概率分配；

+ 分别与encoder的v值相乘，并相加后，得到针对各单词的加权求和值z，即是self-attention的输出。

**所谓的self-attention，是指 所有的keys，values和queries都来自encoder中上一层的输出，encoder中的每一个position都可以处理encoder上一层的所有位置**。

![computations_of_self-attention2](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computations_of_self-attention2.jpg)

![multi-head_self-attention1](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_self-attention.jpg)
![multi-head_self-attention2](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_self-attention2.jpg)
![multi-head_self-attention3](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/multi-head_self-attention3.jpg)

+ decoder

![computation_flow_of_decoder1](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computation_flow_of_decoder.jpg)
![computation_flow_of_decoder2](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/Attention/img/computation_flow_of_decoder2.jpg)

细节动态图[click](https://www.zhihu.com/question/61077555/answer/183884003)

## 5 training
模型训练主要包括：**training data and batching，hardware and schedule，optimizer以及正则化**。

training data：来自标准WMT 2014 english-german dataset，使用byte-pair encoding 编码句子，

使用了3种正则化：residual dropout，label smoothing。

## 6 results
分别在英德机器翻译核英法机器翻译任务中表现优秀。

为评估模型泛化能力，在english constituency parsing任务上进行实验。
## 7 conclusion
第一个提出完全依赖于attention的传到序列模型-transformer；

训练速度显著快于其他基于CNN,RNN的模型。


> references

1. [softmax函数及其导数](https://blog.csdn.net/cassiePython/article/details/80089760)

2. [《Attention Is All You Need 》阅读笔记](https://blog.csdn.net/songbinxu/article/details/80332992)

3. [attention_is_all_you_need解读](https://zhuanlan.zhihu.com/p/34781297)

4. [如何理解谷歌团队的机器翻译新作《Attention is all you need》？](https://www.zhihu.com/question/61077555/answer/183884003)

5. [Transformer模型笔记(包含pytorch代码)](https://zhuanlan.zhihu.com/p/39034683)
