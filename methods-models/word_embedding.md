## 1 word embedding
+ 词编码

**one-hot representation**(独热表征)将单词表示为一个高维稀疏向量；各个单词的表示相互独立。其存在两个问题：①*没有考虑单词间的语义信息*；②向量维度为词表大小，当词表中单词量巨大时，容易*造成维度灾难*，并且计算量将非常大，十分耗时；

**distributed representation**可以解决one-hot representation的问题：通过训练，将高维稀疏的向量 映射到 低维空间中，将每个词映射到一个低维度的语义空间，每个词 将由一个固定维度的稠密向量表示。事实上，distributed representation这个概念在IR领域早就广泛使用，只是被称为*向量空间模型vector space model(VSM)*.VSM基于一种statistical semantics hypothesis：语言的统计特征隐藏着语义的信息。比如，2篇具有相似词分布的文档可以被认为是有着相近的主题。该假设有2个广为人知的衍生版本：一个是bag of words hypothesis：一篇文档的词频代表了文档的主题；一个是distributional hypothesis：上下文环境相似的两个词有着相近的语义。**word2vec是基于distributional hypothesis的**。

**word embeddinig**, 中文又称词嵌入，基于分布式假设(distributional hypothesis:由Harris于1954年提出，假设上下文相似的词，其语义也相似)。         
得名于Bengio等人在2001年的一片论文《neural network language model》，该论文中，模型在学习语言模型的同时，也得到了词向量。下图是谷歌于2013年开源的一款用于计算词向量的工具——word2vec的两种训练方法：

![word_embedding](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/word_embedding.jpg)

woed embedding**是一个映射，将单词从原先所属的空间映射到新的空间中**。同时，通过将word向量化成word vector，可以得到单词间的语义相似性信息。这是因为：向量之间的距离在一定程度上可以衡量词的语义相似性，含义相似的词在空间中的距离更接近。这个是传统词袋模型挖掘不到的信息。
## 2 CBOW model和 Skip-gram model
下图是两种模型的构架图：

![CBOW&Skip-Gram](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/CBOW%26Skip-Gram.png)
> 2.1 **CBOW model**

根据某个词前面的 c个词，或前后c/2个连续的词，找到使用softmax函数的输出层中概率最大的那个词，即是所求的某个词。

在BOW模型中，输入层是上下文的c个词的one-hot vector(每个词都是平等的，即不考虑他们与中心词的距离),输出层是词汇表大小个数量的神经元（即词汇表中所有词的softmax函数概率值）；通过DNN的反向传播算法，求得DNN模型的参数，同时得到词表中所有词对应的词向量；<br>
**针对具体的task（给出上下文词，求中心词），使用训练好的参数和词向量**，通过前向传播算法和softmax激活函数，找到概率最大的词，即是我们输入层的c个词对应的 可能性最大的中心词。**适合小型数据库**。

> 2.2 **skip-gram model**

根据某个词，找到使用softmax函数的输出层中，概率排前n的n个词。<br>
输入层是中心词的词向量，输出层是词汇表大小个数量的神经元（即词汇表中所有词的softmax函数概率值）；通过DNN的反向传播算法，求得DNN模型的参数，同时得到词表中所有词对应的词向量；<br>
**针对具体的task（给出中心词，求上下文词），使用训练好的参数和词向量**，通过前向传播算法和softmax激活函数，找到概率值大小排前n的词，即是 中心词对应的最可能的n个上下文词。**在大型语料上表现较好**。

**由于词表一般在百万级别以上，这意味着 DNN 的输出层softmax函数需要计算词表大小个数量的词的输入概率，计算量很大，处理过程十分耗时**。

## 3 word2vec
在正式介绍word2vec之前，有必要了解一下NNLM。
+ 3.1 NNLM

**NNLM存在的问题**：
>1. 只能处理定长的序列，缺少灵活性；对此，Mikolov等人在2010年提出RNNLM，使用递归神经网络 代替原始模型中的前向反馈神经网络，，并将embedding layer 同RNN中的 hidden layer合并，解决了变成序列的问题
>2. 训练速度太慢。在百万量级的数据集上，使用40个CPU进行训练，需耗时数周才能得到稍微靠谱的结果。此时，
Mikolov发现：
+ 3.2 word2vec

word2vec是谷歌在2013年提出的一种word embedding工具或算法集合，它是一种**从大量文本语料中，以无监督的方式学习语义知识的模型，通过学习文本，使用词向量表征语义信息,可以较好地表达不同词之间的相似和类比关系**。其实是一个简化的神经网络：input layer为one-hot vector；hidden layer为线性单元，没有使用激活函数；output layer使用softmax函数，维度与input layer一样；采用2种模型（CBOW 和 skip-gram）和2种方法（negative sampling 和 分层softmax）的组合。
+ 以下所讲均**以Skip-Gram模型为例**

word2vec建模过程与自编码器(auto-encoder)的思想相似，**其思想为：先基于训练数据构建一个神经网络，这是一个fake task；然后获取训练好的模型的参数(例如隐层的权重矩阵)。**基于这种思想的建模被称为“fake task”，因为我们没有使用训练好的模型处理新的任务，而只是需·要模型学得的参数。

建模过程分为2各部分：**1.建立模型；2.通过模型获取嵌入词向量**。
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


参考文献:[word2vec的前世今生](https://www.cnblogs.com/iloveai/p/word2vec.html)中又十分详细的讲解！！


