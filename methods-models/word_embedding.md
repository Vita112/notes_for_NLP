## 1 word embedding
+ 词编码

**one-hot representation**(独热表征)将单词表示为一个高维稀疏向量；各个单词的表示相互独立。其存在两个问题：①*没有考虑单词间的语义信息*；②向量维度为词表大小，当词表中单词量巨大时，容易*造成维度灾难*，并且计算量将非常大，十分耗时；

**distributed representation**可以解决one-hot representation的问题：通过训练，将高维稀疏的向量 映射到 低维空间中，将每个词映射到一个低维度的语义空间，每个词 将由一个固定维度的稠密向量表示。事实上，distributed representation这个概念在IR领域早就广泛使用，只是被称为*向量空间模型vector space model(VSM)*.VSM基于一种statistical semantics hypothesis：语言的统计特征隐藏着语义的信息。比如，2篇具有相似词分布的文档可以被认为是有着相近的主题。**该假设有2个广为人知的衍生版本**：一个是bag of words hypothesis：一篇文档的词频代表了文档的主题；一个是distributional hypothesis：上下文环境相似的两个词有着相近的语义。**word2vec是基于distributional hypothesis的**。

**word embedding**, 中文又称词嵌入，基于分布式假设(distributional hypothesis:由Harris于1954年提出，假设上下文相似的词，其语义也相似)。         
得名于Bengio等人在2001年的一片论文《neural network language model》，该论文中，模型在学习语言模型的同时，也得到了词向量。

![word_embedding](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/word_embedding.jpg)

woed embedding**是一个映射(见上图)，将单词从原先所属的空间映射到新的空间中**。同时，通过将word向量化成word vector，可以得到单词间的语义相似性信息。这是因为：向量之间的距离在一定程度上可以衡量词的语义相似性，含义相似的词在空间中的距离更接近。这个是传统词袋模型挖掘不到的信息。

## 3 word2vec
在正式介绍word2vec之前，有必要了解一下NNLM。
+ **3.1 NNLM神经网络语言模型**

2003年，由Bengio等人在《a neural probabilisitic language model》中提出，在模型构建过程中产生的映射矩阵，为后续包括word2vec等研究word representation learning奠定了基础。

NNLM基本思想如下：
```
1. 词表中每一个word 都对应 一个连续的特征向量；
2. 存在一个连续平滑的概率模型，输入一段词向量序列，可以输出该序列的联合概率；
3. 同时学习 词向量权重和概率模型的参数。
```
该模型主要有2部分：
> 1. 线性embedding layer：它将input word 的one-hot vector映射为一个 低维稠密的real-value vector；

> 2. 简单的前向反馈神经网络g，由tanh隐层和softmax输出层组成。将经过embedding layer输出的word representation(此时应该是输入词段的向量拼接，假设词段长度为k)映射为长度为词典大小的概率分布向量，对词典中的word在输入context下的条件概率做预估：
$$p(w_{i}|w_{1}\cdots w_{i-1})\approx f(w_{i},w_{t-1},\cdots, w_{t-n+1})\approx g(w_{i},C(w_{t-n+1}),\cdots ,C(w_{t-1}))$$
使用regularized cross-entropy loss function来优化模型参数θ：
$$L(\theta )=\frac{1}{T}\sum_{t}logf(w_{t},w_{t-1},\cdots, w_{t-n+1})+R(\theta )$$
模型参数θ包括了 embedding layer矩阵C的各元素 和 前向反馈神经网络g中的权重。

**模型很简单，却同时解决了两个问题**：
>1. 计算了LM的条件概率p(wt|context);
>2. 学习了input word 的word representation

**NNLM存在的问题**：
>1. 只能处理定长的序列，缺少灵活性；对此，*Mikolov等人在2010年提出RNNLM，使用递归神经网络 代替原始模型中的前向反馈神经网络*，并将embedding layer 同RNN中的 hidden layer合并，解决了变成序列的问题

>2. 参数空间巨大，训练速度太慢。在百万量级的数据集上，使用40个CPU进行训练，需耗时数周才能得到稍微靠谱的结果。此时，**Mikolov发现：可以简化NNLM中的第二步，得到word的连续特征向量**。他于2013年推出2篇paper，并开源了word2vec开元词向量计算工具。
+ **3.2 word2vec**

word2vec是谷歌在2013年提出的一种word embedding工具或算法集合，它是一种**从大量文本语料中，以无监督的方式学习语义知识的模型，通过学习文本，使用词向量表征语义信息,可以较好地表达不同词之间的相似和类比关系**。

word2vec是一个简化的神经网络：input layer为one-hot vector；hidden layer为线性单元，没有使用激活函数；output layer使用softmax函数对输出进行归一化，得到概率分布，维度与input layer一样；对原始NNLM的改造有以下几点：
>1. word2vec删掉了NNLM中前向反馈神经网络中的nonlinear hidden layer，直接将embedding layer 与 softmax layer连接;
>2. 忽略上下文环境的序列信息（此处对应着词袋模型）：输入的所有词向量均汇总到同一个embedding layer；
>3. put feature word into context

采用2种模型（CBOW 和 skip-gram）和2种方法（negative sampling 和 分层softmax）的组合。

**先介绍CBOW model和 Skip-gram model**，下图是两种模型的构架图：

![CBOW&Skip-Gram](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/CBOW%26Skip-Gram.png)
>> **CBOW model**

根据某个词前面的 c个词，或前后c/2个连续的词，找到使用softmax函数的输出层中概率最大的那个词，即是所求的某个词。是从 根据context对target word的预测中，学习到词向量的表达。**针对具体的task（给出上下文词，求中心词），使用训练好的参数和词向量**，通过前向传播算法和softmax激活函数，找到概率最大的词，即是我们输入层的c个词对应的 可能性最大的中心词。**适合小型数据库**。下面考虑多词上下文的CBOW。

+ 1. 前向传播

主要有3个步骤：从input layer到hidden layer，**线性操作**得到隐层结果；从hidden layer到output layer**线性操作**得到输出；使用softmax函数对输出归一化，得到概率分布（设词表大小为V）。模型中，输入层是上下文的c个词的one-hot vector(每个词都是平等的，即不考虑他们与中心词的距离),输出层是词汇表大小个数量的神经元（即词汇表中所有词的softmax函数概率值）；

设x1,x2,……,xC是上下文单词的one-hot编码，**以下通过公式展示上面的三个步骤**：
$$h=\frac{1}{C}W^{T}(x_{1},x_{2},\cdots ,x_{C})=\frac{1}{C}(v_{w_{1}}+v_{w_{1}}+\cdots +v_{w_{C}})^{T}\\\\
u=W'^{T}h,u_{j}=v'\_{w_{j}}^{T}h$$
由于`h是输入单词的词向量`，而$v'\_{w_{j}}^{T}$是第j个单词的词向量，**因此，$u_{j}=v'\_{w_{j}}^{T}h$可以看做是 输入单词的词向量与第j个单词词向量的相似度**。
$$y_{j}=\frac{exp(u_{j})}{\sum_{j'=1}^{V}exp(u_{j'})}=p(w_{j}|w_{I}),j=1,\cdots ,V$$
+ 2. 反向传播

主要有3点：①最大似然估计和损失函数；②hidden layer到output layer的权重矩阵W'的更新；③input layer到hidden layer的权重矩阵W的更新

>  MLE 和 loss function

MLE:
$$MaxP(w_{O}|w_{I,1},w_{I,2},\cdots ,w_{I,C})=MaxlogP(w_{O}|w_{I,1},w_{I,2},\cdots ,w_{I,C})$$

loss function：
$$E=-logP(w_{O}|w_{I,1},w_{I,2},\cdots ,w_{I,C})\\\\=-u_{j^{\*}}+log\sum_{j'=1}^{V}exp(u_{j'})$$
> 使用SGD，更新hidden layer到output layer的权重矩阵W'，求导过程如下：

$$\frac{\partial E}{\partial u_{j}}=-\frac{\partial u_{j^{\*}}}{\partial u_{j}}+\frac{\partial log\sum_{j'=1}^{V}exp(u_{j'})}{\partial u_{j}}=y_{j}-t_{j}:=e_{j}$$
其中，$t_{j}$为指示函数，
$$t_{j}=\left\{\begin{matrix}
1, &j=j^{\*}\\ 
0, &j\neq j^{\*}
\end{matrix}\right.$$
此处，yj-tj可看做 预测概率与真实概率的差值，即误差ej。接下来，计算权重矩阵W'的第i行第j列的元素$w'\_{ij}$的梯度为：
$$\frac{\partial E}{\partial w'\_{ij}}=\frac{\partial E}{\partial u_{j}}\times \frac{\partial u_{j}}{\partial w'\_{ij}}=e_{j}h_{j}$$

此处，hj是向量h的第j个分量。于是梯度更新公式为：
$$w'\_{ij}^{(new)}=w'\_{ij}^{(old)}- \eta e_{j}h_{j}$$
或者为
$$v'\_{wj}^{(new)}=v'\_{wj}^{(old)}- \eta e_{j}h_{j}$$
$$v'\_{wj}$是矩阵W'的第j列。计算ej时，当j≠j\*时，tj=0，即ej=yj-tj≥0，则会 减去ηej倍的h；当j=j\*时，tj=1，则ej=yj-tj≤0，W'中对应的列向量会 增加ηej倍的h。

> 使用SGD计算 input layer到hidden layer的权重矩阵的W的更新

计算损失函数E对隐层向量h的偏导：
$$\frac{\partial E}{\partial h}=\sum_{j=1}^{V}\frac{\partial E}{\partial u_{j}}\times \frac{\partial u_{j}}{\partial h}\\\\=\sum_{j=1}^{V}e_{j}\frac{\partial (v{\'}\_{wj}^{T}h)}{\partial h}=\sum_{j=1}^{V}e_{j}v{\'}\_{wj}^{T}:=EH $$
上式可看作是 以ej为权重，对所有W'的列向量进行加权求和。接上式，求关于W的偏导：
$$\frac{\partial E}{\partial W}=\frac{\partial E}{\partial h}\frac{\partial h}{\partial W}\\\\
=x\cdot EH^{T}$$
$EH^{T}$是 1×N 的行向量，输入向量x为 V×1 .因此上式得到一个 V×N 的矩阵。又 由于x是one-hot vector，故得到的其实是向量x中 分量不为0的位置对应的$EH^{T}$的值。此时梯度更新公式为：
$$v_{wI,c}^{(new)}=v_{wI,c}^{(old)}-\frac{1}{C}\eta EH^{T}$$
其中，$v_{wI,c}$ 表示 输入单词在矩阵 W 中对应的行向量，此处 需要分别对上下文中的多个词向量进行更新。

>> **skip-gram model**

根据某个词，找到使用softmax函数的输出层中，概率排前n的n个词。**从 根据given word对context的预测中，学些到word representation**。
+ 1. 前向传播

对应CBOW中， hidden layer到output layer计算后，同样使用softmax计算概率分布，**skip-gram会输出多个概率分布，假设有C个分布，则第c个分布的第j个输出，即第c个上下文单词为j的概率为**：
$$p(w_{c,j}|w_{I})=\frac{e^{u_{c,j}}}{\sum_{j'=1}^{V}e^{u_{j'}}}=y_{c,j}$$
其中，$w_{c,j}$表示第c个概率分布的第j个单词，$y_{c,j}$表示 第c个概率分布中第j个单词的输出概率。

由于skip-gram中，不同的概率分布使用了相同的输出矩阵W'，因此，对于不同的c在相同位置上的u是相等的，即
$$u_{c,j}=u_{j}=v'\_{wj}^{T}\cdot h$$
其中，$v'\_{wj}$是 词典中第j个单词wj的输出向量，也是矩阵W'的第j列。
+ 2. 反向传播

> 损失函数

$$E = -logP(w_{O,1},\cdots ,w_{O,C}|w_{I})\\\\
=-log\prod_{c=1}^{C}P(w_{O,C}|w_{I})\\\\
=-log\prod_{c=1}^{C}\frac{exp(u_{c,j_{c}^{\*}})}{\sum_{j'=1}^{V}exp(u_{j'})}\\\\
=-\sum_{c=1}^{C}u_{c,j_{c}^{\*}}+C\cdot log\sum_{j'=1}^{V}exp(u_{j'})$$
其中，$j_{c}^{\*}$是第c个上下文真实单词在字典中的索引。
> hidden layer 到output layer的权重矩阵W'的更新

$$\frac{\partial E}{\partial u_{c,j}}=y_{c,j}-t_{c,j}:=e_{c,j}$$
由于输出层权重共享，即$y_{c1,j}=y_{c2,j}$。因此，只是$t_{c,j}$的不同，会导致$e_{c,j}$的不同。于是，定义
$$EI_{j}=\sum_{c=1}^{C}e_{c,j}$$
定义V维向量
$$EI=\[EI_{1},EI_{2},\cdots ,EI_\{V}]$$
计算损失函数E关于W'的导数：
$$\frac{\partial E}{\partial w'\_{ij}}=\sum_{c=1}^{C}\frac{\partial E}{\partial u_{c,j}}\cdot =\frac{\partial u_{c,j}}{\partial w'\_{ij}}= EI_{j}\cdot h_{j}$$
因此，权重更新公式为：
$${w}'\_{ij}^{(new)}={w}'\_{ij}^{(old)}-\eta EI_{j}\cdot h_{j}$$
or
$${v}'\_{wj}^{(new)}={v}'\_{wj}^{(old)}-\eta EI_{j}\cdot h$$
> input layer到hidden layer的权重矩阵W的更新

$$\frac{\partial E}{\partial h}=\sum_{j=1}^{V}\sum_{c=1}^{C}\frac{\partial E}{\partial u_{c,j}}\cdot \frac{\partial u_{c,j}}{\partial h}\\\\
=\sum_{j=1}^{V}\sum_{c=1}^{C} e_{c,j}\ {v'}\_{wj}^{T}\\\\
=\sum_{j=1}^{V}{v'}\_{wj}^{T}\sum_{c=1}^{C} e_{c,j}\\\\
=\sum_{j=1}^{V}EI_{j}{v'}\_{wj}^{T}:=EH$$

观察上式发现：EH是输出词向量${v'}\_{wj}^{T}$的加权求和，权重由$EI_{j}=\sum_{c=1}^{C}e_{c,j}$给出。由于skip-gram需要预测多个上下文，于是，预测误差是 将多个上下文在同一个位置的预测误差相加得到。

**其本质是 计算word 的input representation 和 目标representation之间的余弦相似度，并进行softmax归一化**。通过DNN的反向传播算法，求得DNN模型的参数，同时得到词表中所有词对应的词向量；<br>
**针对具体的task（给出中心词，求上下文词），使用训练好的参数和词向量**，通过前向传播算法和softmax激活函数，找到概率值大小排前n的词，即是 中心词对应的最可能的n个上下文词。**在大型语料上表现较好**。

**由于词表一般在百万级别以上，这意味着 DNN 的输出层softmax函数需要计算词表大小个数量的词的输入概率，计算量很大，处理过程十分耗时**。因此，Mikolov引入了两种优化算法：**分层softmax(hierarchical softmax)和负采样(negative sampling)**.

>> Hierarchical softmax

最早由 bengio于05年引入语言模型中。其**基本思想是：将复杂的归一化概率分解为一系列条件概率的乘积**：
$$p(v|context)=\prod_{i=1}^{m}p(b_{i}(v)|b_{1}(v),\cdots ,b_{i-1}(v),context)$$
其中，每一层条件概率对应一个二分类问题，通过一个logistic regression function拟合。于是，将对V个word的概率归一化问题 转化成了 对V个word的log似然概率进行拟合。

层次softmax 通过构造一颗二叉树，将目标概率的计算复杂度 从最初的V降低到了logV，即计算一次概率，最坏要跑O(logV)个节点。其付出的代价是：人为增强了词与词之间的耦合性。比如，一个word出现的条件概率的变化，会影响到其二叉树路径上所有非叶节点的概率变化，间接对 其他word出现的条件概率产生不同程度的影响。此外，如果训练样本里的中心词w是一个很生僻的词，则 需要在霍夫曼树中辛苦地乡下走很久。

*实际应用中，基于Huffman编码的二叉树可以满足大部分应用场景的需求*。<br>
因为①Huffman树 是满二叉树，从BST 角度上讲，平衡性最好；②Huffman树 可以构成优先队列，对非随机访问十分有效。因此，**按照词频降序建立Huffman树，保证了高频词接近Root，也就是说，高频词计算少，低频词计算多，是一种贪心优化算法**。
>> negative sampling

是Tomas Mikolov 等人在论文《distributed representations of words and phrases and their compositionality》提出的，**是NCE(Noise Contrastive Estimation)噪声对比评估的简化版，目的是 提高训练速度，改善所得词向量的质量**。

关于**negative sampling的直观理解**：假设有一个训练样本，中心词为w，其上下文为context(w),这是一个正实例；在nagative sampling中，我们得到neg个和w不同的中心词wi=1,2，……，neg,context(w)和wi就组成了neg个负实例。使用正实例(w,context(w))和neg个负实例(wi,context(w))进行二元逻辑回归，得到负采样对应每个词wi对应的模型参数Θi 和 每个词的词向量。

> NCE本质是 利用已知的概率密度函数来估计未知的概率密度函数，使得在 没法直接完成归一化因子(配分函数)的计算时，能够估算出概率分布的参数。其思想如下：

在softmax回归中，计算 某个样本属于某个分类的概率，需要把所有分类的概率都计算出来。**NCE将多分类变为二分类，并使用相同的参数**：从真实的数据中抽取样本X=（x1,x2,……,xTd），但我们并不清楚 该样本服从何种分布。假设每个样本xi服从一个未知的概率密度函数pd，现在需要一个可参考的分布，也可称为噪音分布来反过来估计概率密度函数pd,从该噪音分布中抽取的样本数据为Y=（y1,y2,……,yTn）。我们的目的是  通过学习一个分类器把这两类样本区别开来，从模型中学到数据的属性，通过比较来学习。

+ 以下所讲**以Skip-Gram模型为例**

word2vec建模过程与自编码器(auto-encoder)的思想相似，**其思想为：先基于训练数据构建一个神经网络；然后获取训练好的模型的参数(例如隐层的权重矩阵)。**实际上，构建构建神经网络是一个fake task，因为我们没有使用训练好的模型处理新的任务，而只是需要模型学得的参数。建模过程分为2各部分：**1.建立模型；2.通过模型获取嵌入词向量**。

> 无监督特征学习中最常见的自编码器：通过在隐层将输入进行编码压缩；然后在输出层将数据解码恢复初始状态；训练完成后，将输出层“去掉”，仅保留隐层。

假设我们有个句子“the dog barked at the mailman”，下面讲一下 skip-gram的主要步骤：
> 1. 从句子中定义一个中心词作为input word。此例中，input word=dog；
> 2. 定义skip-window参数，用于表示从当前input word 的一侧选取词的数量；根据input word 和 skip-window ，构建窗口列表；
此例中，skip_window=2,于是我们最终获得窗口中的词为\[the, dog, barked, at]，整个窗口大小span=4
> 3. 定义num-skips参数，表示从当前窗口选择多少个不同的词作为output word。
此例中，num_skips=2,我们将得到2组(input word, output word)形式的训练数据，即(dog, barked), (dog, the).
> 4. 神经网络基于这些训练数据，输出一个概率分布，**这个概率分布代表着词表中的每个词是output word的可能性**。
结合例子来理解这句话：在步骤3中，我们得到了2组训练数据；假设我们使用(dog, barked)来训练神经网络，那么，
模型通过学习这个训练样本，会告诉我们词表中的每个单词有多大可能性是“barked”，即词表中的每个词有多大可能性跟input word同时出现。

另一个例子帮助理解神经网络的训练过程：

![word2vec_with_skip-gram](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/word2vec_with_skip-gram.png)

给定句子“the quick brown fox jumps over the lazy dog”, window_size=2, 蓝色代表input word，
方框内代表位于窗口内的单词。**模型将从 每对单词出现的次数中 习得统计结果**。
+ 模型细节

**模型中有几个参数比较重要：中心词、窗口大小、移动步长**；<br>
模型的**目标**是：学习 从input layer到hidden layer的 weights matrix.

具体地，首先基于训练文档构建词汇表vocabulary，假设词表中总共有10000个唯一不重复的word，
> 1. 在input layer，使用one-hot encoding表征input word wi，word wi是一个10000维的向量；
> 2. 在hidden layer，学习weights matrix表征word embedding：
设定模型调优后的超参为300(意味着每个单词可以被表示为一个300维的向量)，则权重矩阵$\mathbf{W}\in \mathbf{R}^{10000\times 300}$，
every line is a low-dimensional real-velue vector for the corresponding word.我们使用 wi·W 得到 单词wi的词嵌入向量；


参考文献:

[word2vec的前世今生](https://www.cnblogs.com/iloveai/p/word2vec.html)中又十分详细的讲解！！

[CBOW和Skip-gram模型原理推导](https://blog.csdn.net/bqw18744018044/article/details/90295730)

[噪音对比估计NCE](https://blog.csdn.net/littlely_ll/article/details/79252064)

[噪声对比估计杂谈：曲径通幽之妙](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/80731084)

[word2vec原理（三）基于negative sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html）


