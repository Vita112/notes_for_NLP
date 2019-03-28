title：neural relation extraction with selective attention over instances(基于选择性注意力模型的多实例神经关系抽取)

清华 Lin et al. 2016
## abstract
+ propose a sentence-level attention-based model 
> **first**, use CNN to embed semantics of sentences; **then**,bulid sentence-level attention over multiple instances.

+ contributions:

make full use of all informative sentences of each entities; 

reduce the influence of wrong labelled instances effectively.
+ souce code: https://github.com/thunlp/NRE.
## 1 introduction and related work
KBs: mostly compose of relational facts with triple format, like(Microsoft, founder, Bill Gates); incomplete to contain infinite real-world facts.
> to enrich KBs, many efforts have been invested in automatically finding unknown relational facts.**RE**: the process of generating
relational data from plain text.
### 1.1 RE with distant supervision
+ **Mintz et al. 2009** : 《distant supercision for relation extraction without labeled data》
> propose **distant supervision** to automatically generate training data **via aligning KBs and texts**.     
do not need to manually design features, **but suffer from wrong labelling problem**.
+ **Riedel et al.2010, Hoffmann et al.2011, Surdeanu et al.2012**: adopt multi-instance learning to address wrong labellinig problem.**that is because multi-instance learning consider the reliability of the labels for each instance**.

《modeling relations and their mentions without labeled text》:multi-instance single-label learning

《knowledge-based weak supervision for information extraction of overlapping relations》:multi-instance multi-label learning

《multi-instance multi-label learning for relation extraction》:multi-instance multi-label learning
 
> using NLP tools, like POS tagging, lead to **error propagation**. Because the accururacy of syntactic parsing decrease significantly 
with increasing sentence length.
+ **Socher et al.2012, Zeng et al.2014, dos Santos et al.2015, Zeng et al.2015**: employ deep learning methods

《semantic compositonality through recursive matrix-vector spaces》: use RNN to automatically learn featuers

《relation classification via convolutional deep neural network》:adopt end-to-end CNN

《classifying relations by ranking with convolutional neural networks》:adopt end-to-end CNN

《distant supervison for relation extraction via PCNN》:apply PCNNs to capture structural information between 2 entities;
employ multi-instance learning to address wrong label problem

> base on a sumption:提及这2个实体的句子中，至少有一个句子表达了他们的关系，而且在训练时，针对每一对实体，仅选择那个最可能的句子，来进行预测。**这导致：模型会丢失掉 包含在未观察到的句子中的 大量丰富有用的信息**。

### 1.2 what's new in this paper?

![architecture_of_sentence-level_attention-based_CNN]()

as described in Fig.1, mainly contain 3 steps:
> 1. use CNN to embed the senmatics of sentences
> 2. represent the relation as semantic composition of sentence embeddings（将关系表征为句子嵌入的语义复合）
> 3. build sentence-level attention over multiple instances,and extract relation with the relation vector
weighted by sentence-level attention.

### 1.3 attention model
[Reference blog1](https://blog.csdn.net/malefactor/article/details/78767781)

[Reference blog2](https://blog.csdn.net/mpk_no1/article/details/72862348)
  
[attention_model](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/attention_model.md)

## 2 Methodology
mainly contains 2 parts:
> 1. **sentence encoder**:给定一个句子x和2个目标实体，使用CNN构建句子的分布式表征X；
> 2. **selective attention over instances**:当学习了所有句子的分布式向量表征后，使用sentence-level attention 来选择真正表达对应关系的句子。
### 2.1 Sentence Encoder
![sentence_encoder_in_selective_attention_with_CNN]()

**Transform the sentence x into its distributed representation X By CNN/PCNN**.
+ **Input representation**

use **word embedding** to transform words into distributed representations(low-dimensional vectors) to capture syntactic and semantic 
meanings of the words.
> word embeddinig：中文又称词嵌入，基于分布式假设(distributional hypothesis:由Harris于1954年提出，假设上下文相似的词，其语义也相似)。            得名于Bengio等人在2001年的一片论文《neural network language model》，该论文中，模型在学习语言模型的同时，也得到了词向量。

> 1. 神经网络语言模型大致有：① Neural netword language model; ② Log-bilinear language model; ③ Recurrent neural network based language model;④ C&W model; ⑤ CBOW 和 skip-gram model
> 2. **CBOW model**：根据某个词前面的 c个词，或前后c/2个连续的词，找到使用softmax函数的输出层中概率最大的那个词，即是所求的某个词。在BOW模型中，输入层是c个词的词向量(每个词都是平等的，即不考虑他们与中心词的距离),输出层是词汇表大小个数量的神经元（即词汇表中所有词的softmax函数概率值）；通过DNN的反向传播算法，求得DNN模型的参数，同时得到词表中所有词对应的词向量；针对具体的task（给出上下文词，求中心词），使用训练好的参数和词向量，通过前向传播算法和softmax激活函数，找到概率最大的词，即是我们输入层的c个词对应的 最可能的中心词。
> 3. **skip-gram model**：根据某个词，找到使用softmax函数的输出层中，概率排前n的n个词。输入层是中心词的词向量，输出层是词汇表大小个数量的神经元（即词汇表中所有词的softmax函数概率值）；通过DNN的反向传播算法，求得DNN模型的参数，同时得到词表中所有词对应的词向量；针对具体的task（给出中心词，求上下文词），使用训练好的参数和词向量，通过前向传播算法和softmax激活函数，找到概率值大小排前n的词，即是 中心词对应的最可能的n个上下文词。
下面讲一下 skip-gram的主要步骤：
> 1. 从句子中定义一个中心词，即input word；
> 2. 定义skip-window参数，用于表示从当前input word 的一侧选取词的数量,即找到最有可能是中心词周围的词的个数；
> 3. 根据input word 和 skip-window ，构建窗口列表；
> 4. 定义num-skips参数，表示从当前窗口选择多少个不同的词作为output word。

> **模型中有几个参数比较重要：中心词、窗口大小、移动步长**；模型的**目标**是：学习 从input layer到hidden layer的 weights matrix.

具体地，在input layer，使用one-hot representation表征input word wi；

在hidden layer，学习weights matrix表征word embedding：假设词表中总共有10000个word，设定模型调优后的超参为300，则权重矩阵W∈$R^10000×300$，every line is a low-dimensional real-velue vector for the corresponding word.我们使用 wi·W 得到 单词wi的词嵌入向量；



**use word2vec tool for implementation**.word2vec是谷歌在2013年提出的一种word embedding工具或算法集合，采用2种模型（CBOW 和 skip-gram）和2种方法（negative sampling 和 分层softmax）的组合。
