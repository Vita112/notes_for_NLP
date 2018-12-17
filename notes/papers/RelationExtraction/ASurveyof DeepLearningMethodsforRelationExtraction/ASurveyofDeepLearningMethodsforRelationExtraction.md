比较不同DL模型在RE任务中的表现
## 1 introduction
在监督方法中，关系抽取和分类任务指的是，使用含有实体对mentions的文本，将实体对分类为一组已知关系。RE具体指，预测一个给定文本中的实体对是否包含一个关系。RC具体指，假定文档包含一个关系，则预测该文档指向的给定本体中的哪个关系类。加上一个额外类NoRelation，这两个task可以结合为一个multi-class 分类问题。
传统监督方法中，RE方法通常由两类:`feature-based method和kernel-based method`。2种方法都需要大量训练集用于学习，需要人工标注语料，且对语料标注精确度要求较高。
### 1.1 datasets
以下数据样本中，文档句子已标注好命名实体，且实体对间的关系类别已被预测。
+ ACE 2005 dataset:Automatic Context Extraction.包含599个文档，有7个主要关系类型。
+ SemEval-2-10 task 8 dataset：包含10717个样本，有9个有序关系类型，即2\*9+1=19个关系类型.
### 1.2 distance supervision
+ Mintz2009基于一个假设：如果在KB中的一个实体对间存在一个关系，那么包含这个实体对的mention的每一个文本都有这种关系。**但是**，我们知道`并不是每一个包含了实体提及的文档中，一定存在一个关系`。于是，Riedel 2010使用纽约时报语料库来对其FREEBASE的关系，有53中可能的关系类，
+ word enbedding
是一个单词在词汇中的分布表征形式，每一个单词表示为 一个低维空间的一个向量。词嵌入旨在捕获单词的句法和语义信息。
+ positional embeddings
使用positional embeddings，模型可以编码 输入在句子中每一个单词到实体的相对距离。(Zeng 2014)有一个想法是：更靠近目标实体的单词，通常包含更多有关关系类的有用信息。
+ convolutional neural networks
为更进一步编码句子，re的深度学习模型通常使用卷积神经网络类捕获n-gram level 的特征。卷积层操作如下：
```
1 给出一个输入句子x，其向量序列为x = {w1,w2,……,wm},如果l是卷积层滤波器的窗口大小，则，第i个窗口的向量由连接那个窗口的输入向量形成：
qi = wi:i+l-1
一个单独的卷积核(滤波器)由一个权重向量W，一个偏置项b组成，则，第i个窗口的输出将这样计算：
pi = f(W'qi+b),其中，f是激活函数。
```
### 2 supervised learning with CNNs
#### 2.1 simple CNN model(Liu et al,2013)
可能是最早使用CNN来自动学习特征的一个尝试，他首先使用word vector 和 lexical features 对输入句子进行编码，建立了一个end-to-end network，使用一个卷积核层，单层神经网络和softmax output layer，来对整个所有的关系类别给出概率分布。*使用同义向量代替词向量，为每一个同义类分配一个向量*。**然而**，它未能利用词嵌入的真正表现力。嵌入不是在语料库上以无监督的方式训练，而是随机分配给每个同义词类。此外，该模型还尝试使用单词列表、POS列表和实体类型列表合并一些词汇特性。**即便如此，该模型在ACE2005数据集上的性能表现比最先进的基于内核的模型要高9分**。
#### 2.2 CNNmodel with max-pooling(Zeng et al,2014)
使用了 预先在一个大型未标注语料库中训练好的 word embeddings；第一个使用 positional embeddings；使用一些 比如有关名词的信息等 词汇级别特征，和名词的WordNet上位词。**一个重要贡献是**，在卷积网络的输出上使用最大聚汇层max-pooling layer。通过使Z折叠为Z'进行最大化操作，Z'的维度独立于句子的长度m。
#### 2.3 CNN with multi-sized window kernels(Nguyen and Grishman, 2015)
模型完全放弃了使用外部词汇特征，来丰富输入句子的表征，而是让CNN自己学习所需要的特征。模型包含词嵌入，位置嵌入，然后是卷积和最大池化。**此外**，它们结合了不同窗口大小的卷积滤波器，来捕获更广范围的n-gram 特征。作者使用word2vec训练的 预训练的word embeddings 来初始化 word embedding metrix。
### 3 multi-instance learning models with distant supervision
multi-instance learning是一种监督学习的形式，其中一个标签被给予一堆实例，而不是一个实例。，每一个实体对定义一个袋子，袋子由所有包含该实体对提及的句子组成。**标签被给予每个关系实体的袋子**。这个模型基于这样一种假设：*如果一个关系存在一个实体对之间，那么在这个实体对袋子中，至少有一个文档必须要反映这个关系*。
#### 3.1 piecewise convolutional neural networks(Zeng et al 2015)
**一个重要贡献是**，通过在句子的不同分块中进行最大池化，即分段最大池化操作，来避免 在整个句子中进行最大池化操作时遇到的问题：不足以捕获整个句子中实体间的结构。作者称，基于聚焦的2个实体的位置，每一个句子可以被自然地切分为3块。**一个缺点**：在损失函数中设置多实例地方法 有问题。模型的loss funtion定义如下：
>· 给定T袋文档，每一袋包含qi个文档，文档标签表示为yi，i=1,2,……,T；
>· 神经网络给出从袋子i的文档j中抽取关系r的概率，公式为：
>$p(r|d^{j}_{i},\theta); j=1,2,……,qi$

其中西θ是神经网络的加权参数，损失函数为
$$ J(\theta )=\sum_{I=1}^{T}logp(y_{i}|d_{i}^{j*},\theta )$$

$j*=argmax_{j}p(y_{i}|d^{j}_{i},\theta )$;j=1,2,…… ,$q_i$

**但是**，由于在训练和预测过程中，模型**只使用最可能文档的实体对**，这意味着：**模型忽略了大量有用的，由袋子中其他的句子表达的数据和信息**。此问题在下一种方法中得到解决。
#### 3.2 selective attention over instance(Lin et al 2016)
