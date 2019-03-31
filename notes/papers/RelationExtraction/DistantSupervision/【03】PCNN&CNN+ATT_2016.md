title：neural relation extraction with selective attention over instances(基于句子级别选择性注意力模型的多实例神经关系抽取)

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

《multi-instance multi-label learning for relation extraction》:multi-instance multi-label learning + 贝叶斯网络
 
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

![architecture_of_sentence-level_attention-based_CNN](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/architecture_of_sentence-level_attention-based_CNN.jpg)

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

for more information, refers to [word_embedding](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/word_embedding.md)

**use word2vec tool for implementation**.

**position embedding**:the words close to the target entities are usually informative to determine the relaiton between entities.**word embedding specified by entity pairs can help CNN to keep track of how close each word is to head or tail entities**.

as is shown in Fig.2, **every word is transformed to vector representation**, which concatenates word embedding and position embedding.
+ **convolution, max-pooling and non-linear layers**

*because the length of the sentences is variable and the important information can appear in any area of the sentences*, 
we **use CNN to merge all the local featrues**.concretely，
> 1. convolutional layer extract local feature with a sliding window of length l over the sentence,for example,
the i-th filter of convolutional layer is described as $p_{i}=\[\mathbf{W} q+b]\_{i}$ mathematically, here $\mathbf{W}\in \mathbf{R^{d^{c}\times (l\times d)}}$, $q_{i}\in \mathbf{R}^{l\times d}$, $q_{i}$ is the concatenation of a sequence of w word embeddings.
> 2. combine all the local features via a max-pooling operation to obtain a fixed-sized vector for the input sentence;
the i-th element of the vector x : $\[x]\_{i}=max(p_{i})$ ;
**apply PCNN to divide each convolutional filter $p_{i}$ into 3 segments($p_{i1},p_{i2},p_{i3}$) by head and tail entities**.
so the max pooling procedure is performed in 3 segments separately,defined as:
$$\[x]\_{ij}=max(p_{ij})$$
> 3. apply a non-liear function at output, such as the hyperbolic tangent(双曲正切函数)。

after done these, we **got the representation for sentences**, 每个句子表征xi包含 句子中的实体对是否存在一种关系的 信息。
### 2.2 selective attention over instances
对每个句子xi赋予不同的权重αi，得到句子集合S的加权表示向量s；通过softmax函数，输出某个关系r在某个加权句子集合向量s出现 与 
所有关系在某个加权句子集合向量s中出现 的概率大小。

a set $\mathbf{S}$ contains n sentences for entity pair(head, tail), $\mathbf{S}=\{x_{1},x_{2},\cdots ,x_{n}}$.
**our model represents the set $\mathbf{S}$ with a real-valued vector s when predicting relation r.The representation of set $\mathbf{S}$ depends on all sentences' representation x1,x2,……,xn, each sentence representation xi contains information about whether entity pair(head, tail) contains relation r**. SO the setvector s is computed as follows:
$$s=\sum_{i}\alpha \_{i}x_{i}$$, here αi is the weight of each sentence vector xi.**$\alpha \_{i}$ is defined in 2 ways**:
> 1. *AVERAGE*: 假设在集合X中的所有句子 对集合的表征具有相同的贡献，这意味着 集合S的嵌入表示 是所有句子向量的平均值：
$$s=\sum\_{i}\frac{1}{n}x_{i}$$
> 2. *selective attention*: 如果我们将所有句子平等看待，在训练和测试阶段，错误标注问题将带来大量的噪音。因此，**使用一个选择注意力机制，
来弱化噪音句子**，此时αi被定义为：
$$\alpha \_{i}=\frac{exp(e_{i})}{\sum \_{k}exp(e_{k})}$$,
其中，ei指的是 评分输入句子xi和预测关系r的匹配度的 基于查询的函数。我们使用双线性形式，在不同的方案中实现最佳性能：
$$e_{i}=x_{i}Ar$$,
其中，A是一个加权对角矩阵，r是一个 与关系r相关联的 查询向量，它代表了关系r的表征。

最后，通过一个softmax layer
定义条件概率$p(r|\mathbf{S},\theta )$:
$$p(r|\mathbf{S},\theta )=\frac{exp(o\_{r})}{\sum_{k=1}^{n_{r}}}exp(o\_{k})$$
其中，nr指关系类总数，o是神经网络的最终输出，它对应于所有关系类型的得分：
$$o=\mathbf{M}s+d$$
M代表 关系的表征矩阵。Zeng et al.,2015的论文中的模型，其实是本文选择性注意力的一个特例：Zeng 将拥有最大概率的 句子的权重设为1，把其他的设为0.
### 2.3 optimization and implementation details
+ **Optimization** 
> 1. define **loss function** using cross-entropy at the set level:
$$J(\theta )=\sum_{i=1}^{s}logp(r_{i}|\mathbf{S_{i}},\theta )$$
s是句子集的数量，Θ代表模型的所有参数。
> 2. adopt SGD to minimize the loss function, 通过从训练数据集中随机选取一个mini-batch进行迭代，直到收敛。

+ **implementation**
> employ dropout on the output layer to prevent overfitting,因此，神经网络的最终输出，即所有关系类型的得分被改写为：
$$o=\mathbf{M}(s \circ h)+d$$
其中，向量h 由伯努利随机变量组成，概率为p。
## 3 experiments
### 3.1 dataset and evaluation metrics
+ dataset

dataset is generated by aligning Freebase relations with the New York Times corpus,and are divided into 2 parts 
for training and testing respectively.

entity mentions are found using the Stanford named entity tagger.
+ held-out evaluation

compare the relation facts discovered from the test articles with facts in Freebase.

report 聚合曲线 精度/召回率曲线，以及不同数量句子时的精确度(P@N).
### 3.2 experimental settings
+ word embeddings 

**use word2vec to trainig word embedding** on NYT corpus.选取在语料库中出现超过100次的单词作为词表vocabulary。

+ parameter setttings

tune our model **using 3-fold validation on the training set**; use a **grid search to determine the optimal parameters** 
and select learning rate λ for SGD. all parameters used in the experiments are as follows:

![parameter_settings_in_PCNN+ATT](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/parameter_settings_in_PCNN%2BATT.jpg)
### 3.3 effect of sentence-level selective attention
compare different methods through held-out evaluation.

use CNN medel(Zeng et al.2014) and PCNN(Zeng et al. 2015) as sentence encoders. **then compare the performance of 
the 2 different kinds of CNN with sentence-level attention(ATT)**:
> AVE(represents each sentence set as the average vector of sentences inside the set<br>
> ONE(the at-least-one multi-instance learning 
![figure3_aggregate_precision/recall_curves](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/figure3_aggregate_precisionrecall_curves.png)

we find that **CNN/PCNN+ATT can effectively filter out menningless sentences and alleviate the wrong lebeling problem in DSRE**.
### 3.4 effect of sentence number
compare the performance of CNN/PCNN_ONE,CNN/PCNN+AVE,CNN/PCNN+ATT on the entity pairs which have more than one sentence.
3 test setttings:
> 1. ONE: 对每个测试实体对，随机选择1个句子，使用这个句子预测关系；
> 2. TWO: 对每个测试实体对，随机选择2个句子，使用他们进行关系抽取；
> 3. ALL: 使用每个实体对的所有句子，进行关系抽取

**use all the sentences in training; report the P@100, P@200, P@300 and the mean of them for each medel in held-out evaluation**.the P@N for compared models in 3 test settings are shown in the following tabel:
![P@N_for_relation_extraction_in_the_entity_pairs_with_different_nember_of_sentences](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DistantSupervision/pictures/P%40N_for_relation_extraction_in_the_entity_pairs_with_different_nember_of_sentences.png)

the tabel shows that by taking more useful information into account, the relational facts which CNN+ATT ranks higher 
are more reliable and beneficial to relation extraction.
### 3.5 comparison with feature-based approaches
![performance_comparison_of_proposed_model_and_traditional_methods]()

above figure shows that **CNN/PCNN+ATT can learns the representation of each sentences automatically and can express
sentences well**.
## 4 conclusion and future works
CNN/PCNN with sentence-level selective attention proposed in this paper can make full use of informative sentences
and  reduce the influence of wrong labelled instances.

apply this model in other multi-instance learning tasks; incorporate instance-level selective attention model with
other neural networks models for RE.








