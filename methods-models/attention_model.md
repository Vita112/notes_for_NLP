## 1 人类的视觉注意力
借鉴 人类的注意力机制。先从视觉的选择性注意力机制出发：
![视觉注意力机制](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/%E8%A7%86%E8%A7%89%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.jpg)

上图显示了人类视觉接收一幅图时，如何高效分配有限的注意力资源；红色区域表明视觉系统更关注的目标，也可看做是**注意力的焦点**，
对这一区域投入更多的注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息。
## 2 encoder-decoder 框架
目前大多数注意力模型附着在 encoder-decoder框架下，该框架是一种深度学习领域的研究模式，应用于seq2seq问题。编码器和解码器都不是固定的，
可选择的有：CNN/RNN/BiRNN/GRU/LSTM等

在**文本处理领域**，endoder-decoder框架可以直观理解如下：将其看做 处理一个句子（或篇章）生成另一个句子（篇章）的通用处理模型。
编码过程就是 将输入句子看做一个输入序列，把它转换成一个固定维度的稠密向量；解码过程：将之前生成的固定向量再转化为输出序列。
>对于句子（source，target），目标是给定输入句子source，期待通过encoder-decoder框架来生成目标句子target，source和target
可是同一种语言，也可是不同语言。

+ encoder：对输入句子source进行编码，将输入句子通过非线性变换转化为中间语义表示C：
$$\mathbf{C}=\mathbf{F}(x_{1},x_{2},\cdots ,x\_{m})$$

+ decoder：根据句子source的中间语义表示C和之前生成的历史信息
$$y_{1},y_{2},\cdots ,y_{i-1}$$
来生成i时刻生成的单词$y_{i}$,所以
$$y_{i}=\mathbf{G}(\mathbf{C},y_{1},y_{2},\cdots ,y\_{i-1})$$
+ 应用
> 1. source:中文句子；target：英文句子。   →    机器翻译
> 2. source:一篇文章；target：该文章的概括性描述。   →   文本摘要
> 3. source：图片；target：一句描述语，可以描述图片的语义内容

+ encoder-decoder模型的局限性

此处内容其实与下半部分重复。

**最大局限性在于**：编码器 将整个序列的信息压缩进一个固定长度的向量，即语义向量C，它成为编码和解码之间的唯一联系 。但是，①语义向量C无法完全表示整个序列的信息；②先输入内容携带的信息会被后输入内容的信息覆盖掉(稀释掉)，**导致解码的准确度降低**。
## 3 Attention model
本节内容：首先 以MT为例讲解soft attention模型基本原理；然后 抽象出注意力机制的本质思想； 最后 简单介绍self attention。
### 3.1 soft attention 
观察encoder-decoder框架，可以发现：**encoder-decoder框架并没有体现出*注意力模型*，因为decoder非线性函数在生成目标句子的单词时，
不论生成哪个单词，所使用的输入句子source的语义编码C都是一样的，这意味着对于生成某个目标单词yi来说，句子source中的任意单词具有
相同的影响力，*但实际上，在decoder阶段，输入句子source的语义编码C中的每个单词对于生成某个目标单词yi的影响力是不同的*，因此这个
框架中并没有引入注意力机制**。
> 为何要引入注意力机制？
> 未引入attention 的encoder-decoder框架，在处理较短的输入句子时问题不大，**但如果输入句子较长，此时，所有的
语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，这会丢失很多细节信息**。

+ 引入attention机制

引入attention 后，在编码句子语义阶段，针对source每个单词，注意力模型会给不同的单词分配注意力大小，即**原先固定的中间语义
表示C 被替换成 根据当前输出单词来调整加入注意力模型的变化的Ci**,每个Ci对应着不同源语句子单词的注意力分配概率分布。框架图如下：
![引入注意力机制的edcoder-decoder框架](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/%E5%BC%95%E5%85%A5%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9A%84edcoder-decoder%E6%A1%86%E6%9E%B6.jpg)

+ 以中英机器翻译例句“Tom chase Jerry”为例，说明attention-based source单词的中间语义表示：

句子语义编码表征为（因latex公式中似乎不识别中文，故用拼音代替）：
> $$\mathbf{C}\_{tangmu}=g(0.6f_{2}("Tom"),0.2f_{2}("Chase"),0.2f_{2}("Jerry"))$$
> $$\mathbf{C}\_{zhuizhu}=g(0.2f_{2}("Tom"),0.7f_{2}("Chase"),0.1f_{2}("Jerry"))$$
> $$\mathbf{C}\_{jierui}=g(0.3f_{2}("Tom"),0.2f_{2}("Chase"),0.5f_{2}("Jerry"))$$

**f2变换函数 表示encoder对输入英文单词进行某种变换**，若encoder使用RNN模型，则f2函数输出的结果是 某个时刻输入xi后隐层节点的状态值；
**g变换函数 表示encoder根据单词的中间表示 合成整个句子中间语义表示，一般是构成元素的加权求和**，故加入attention机制后，
source输入句子的单词语义表示Ci本质上是一个加权求和函数：
$$\mathbf{C}\_{i}=\sum_{j=1}^{L_{x}}\alpha \_{ij}h_{j}$$

$L_{x}$表示 输入句子source的长度；$\alpha \_{ij}$表示 在target输出第i个单词时，source输入句子中第j个单词的注意力分配系数，hj表示 source输入句子中第j个单词的语义编码.下图更为直观：
![attention-based_source单词语义表示Ci形成过程](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/attention-based_source%E5%8D%95%E8%AF%8D%E8%AF%AD%E4%B9%89%E8%A1%A8%E7%A4%BACi%E5%BD%A2%E6%88%90%E8%BF%87%E7%A8%8B.jpg)

+ 如何获取source输入句子中单词的注意力分配概率值分布？
> RNN作为具体模型的encoder-decoder框架

![RNN-based_encoder-decoder框架下注意力分配概率计算](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/RNN-based_encoder-decoder%E6%A1%86%E6%9E%B6%E4%B8%8B%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%88%86%E9%85%8D%E6%A6%82%E7%8E%87%E8%AE%A1%E7%AE%97.jpg)
**我们的目标：计算输出yi时，输入句子source中的每个单词xi对应于yi的注意力分配概率分布，也可以理解为某个输入句子单词xi和目标生成单词的对齐概率**。于是，我们**通过将target输出句子的i-1时刻的隐层节点状态$H_{i-1}$ 和 输入句子source中每个单词对应的RNN隐层节点状态hj 进行对比，即使用函数F(hj,Hi-1)来获得目标单词yi和每个输入单词对应的对齐可能性。F函数可采取不同的方法。使用softmax函数 对F函数的输出进行归一化处理，得到符合概率分布取值区间的注意力分配概率分布数值**。
### 3.2 essence of Attention medel
将source中的构成元素想象成是 有一系列<key, value>的键值对构成，此时，给定target中某个元素的query，
通过计算query和各个key的相似性，得到每个key对应value的权重系数，然后对value进行加权求和，最终得到
attention数值。所以**本质上，attention机制是通过计算query和key的相似性，对source中元素的value值进行加权求和**。
公式表示如下：
$$att(query,source)=\sum_{i=1}^{L_{x}}sim(que,k_{i}\ast value)$$

**attention的主要思想**是：从大量信息中有选择地筛选出少量重要信息，并聚焦于这些重要信息，
忽略不重要地信息；聚焦的过程体现在权重系数的计算上，权重代表信息的重要性，value值是其对应的信息，
权重越大越聚焦于其对应的value值上。

**可以将attention机制看作一种软寻址(soft addressing)**:将source看作存储器内 存储的内容，
元素由地址key和值value组成；通过query和存储器内的元素地址key的相似性来寻址，寻址时，
**不是只从存储内容中 找出一条内容，而是可能从每个key中取出内容value，value的重要性根据query和key的相似性决定，
然后，对value值进行加权求和，得到最终的value值，即attention值**。
+ attention机制的具体计算过程

![三阶段attention机制计算过程](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/%E4%B8%89%E9%98%B6%E6%AE%B5attention%E6%9C%BA%E5%88%B6%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B.png)
> 1. 计算query和key的相似性；
+ 相似性计算方法：向量点积、向量cosine相似性、引入额外的神经网络来求值.
> 2. 使用softmax函数对相似性结果进行归一化处理，得到权重系数α；
+ 使用softmax对相似性结果进行归一化处理，使其变有 所有元素的权重之和为1 的概率分布.
> 3. 根据权重系数α对value值进行加权求和.

### 3.3 self attention(also called intra attention)
一般任务的encoder-decoder框架中，attention机制发生在target元素query 和 source中的所有元素之间。
**而self attention指的不是source和target之间的attention机制，而是source内部元素之间或target内部元素之间
的attention机制**。
+ self attention机制可以学到哪些规律？以下以机器翻译中的self attention为例说明：
![self_attention可视化实例](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/self_attention%E5%8F%AF%E8%A7%86%E5%8C%96%E5%AE%9E%E4%BE%8B.png)

上图可以看出：self attention可以捕获同一个句子中单词之间的句法 或 语义特征。**引入self attention后，模型更容易
捕获句子中长距离的相互依赖特征，它直接将句子中任意两个单词的联系，通过一个计算步骤直接联系起来，极大地缩短了
远距离依赖特征之间的距离，因此我们得以利用这些特征**。
## 4 Application of Attention model
+ image-caption图片描述

使用encoder-decoder框架来完成目标任务：encoder阶段，使用CNN对图片进行特征抽取；decoder阶段，使用RNN/LSTM输出 对所给图片的等价语义描述。
**加入attention机制后，在输出某个实体单词时，注意力焦点将聚焦在图片中相应的区域上，这个过程
于人类视觉选择性注意机制十分相似**。下图给出了一些例子：
![attention_machanism_in_image_caption_task](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/attention_mechanism_in_image_caption_task.png)

[Reference blog1](https://zhuanlan.zhihu.com/p/37601161)

[Reference blog2](https://blog.csdn.net/mpk_no1/article/details/72862348)

![【paper】show_attend_and_tell-neural_image_caption_generation_with_visual_attention](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/%E3%80%90attention_model%E3%80%91show_attend_and_tell-neural_image_caption_generation_with_visual_attention.pdf)
