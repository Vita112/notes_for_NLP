title: nueral relation extraction with multi-lingual attention 
yankai lin et al. Tsinghua University 2017
## abstruct
propose a multi-lingual nueral relation extraction framework, which process cross-lingual attention to consider the information
consistency and complementary among cross-lingual texts.

source code: https://github.com/thunlp/MNRE
## 1 introduction
人们构建知识库来储存有关真实世界的结构化信息。**the facts in KBs are typically organized in the form of triple**, such as(New York, cityOf, United 
States)。在关系抽取的任务中，远程监督是最具有前途的方法，它通过对齐KBs 和 纯文本 生成训练数据，解决了监督训练数据缺乏的问题。在上一篇论文中，已经详细介绍了
关系抽取任务的基于远程监督的各种方法，包括PCNN, MIML+PCNN, PCNN+ATT等模型，使得关系抽取任务的性能得到极大提升。**但是，大多数关系抽取任务主要关注 
从单语言数据集中抽取关系事实，而实际上，人们使用不同的语言对有关真实世界的知识进行描述；并且由于人类经验和人类认知系统的相似性，说着不同语言的人共享着有关真实世界的
相似知识**。

multi-lingual data will benefit relation extraction for 2 reasons：
> 1. pattern consistency among languages 语言间的模式一致性<br>
在不同的语言中，人们使用固定的模是来表达一个关系事实，并且，各语言中这些模式的对应关系具有一致性
> 2. complementarity 互补性<br>
对于近一半的关系来说，表达这些关系的关系事实的句子的数量 在不同语言中 是不同的。

steps in this paper：
```
1. 使用CNN 将句子中的关系模式 嵌入为一个实值向量；
2. 使用mono-lingual attention 在每个语言中选择 那些富含信息的句子，使用cross-lingual attention 来利用各语言之间的模式一致性和互补性；
3. 我们聚集了 通过mono-lingual attention和 cross-lingual attention加权的所有的句子向量，得到了全局向量，并根据这个全局向量进行关系分类，
```
## 2 related work
与【03】重复部分省略。

Faruqui, Kumar(2015): present a language independent open domain relation extraction system;

Verga et al.(2015): employ Universal Schema to combine OpenIE and link-prediction perspective for multi-lingual RE.

**what's new in this paper**: aim to jointly model the texts in multiple languages to enhangce relation extraction with distant supervision.
## 3 methodology
**key motivation of MNRE**: for each relational fact, the relation patterns in sentences of different languages should be substantially consistent,
and MNRE can utilize the pattern consistency and complementarity among languages to achieve better results for RE.

给定一对实体，他们在m种不同语言中对应的句子被定义为
$$T={S_{1},S_{2},\cdots ,S_{m}}$$
其中，$S_{j}$代表j语言中的包含$n_{j}$个句子的句子集合，
$$S_{j}={x_{j}^{1},x_{j}^{2},\cdots ,x_{j}^{n_{j}}}$$
我们的模型针对每个关系r给出一个分数f(T,r),MNRE框架主要包含2个部分：
> 1. sentence encoder:给定一个句子x和2个目标实体，使用CNN 将句子x中的关系模式编码为一个分布式表示X。
> 2. multi-lingual attention: 当各种语言中的所有句子被编码为分布式表示后，聚集 注意力加权后的所有句子向量，得到全局向量，用于关系预测。
### 3.1 sentence encoder
过程与【03】中的句子编码过程类似，略。
### 3.2 multi-lingual attention
+ mono-lingual attention that selects informative sentences within one language

+ cross-lingual attention that measures the pattern consistency among languages
### 3.3 prediction

## 4 experiments
### 4.1 datasets and evaluation metrics
### 4.2 experimental settings
### 4.3 effectiveness of consistency
### 4.4 effectiveness of complementarity
### 4.5 comparison of Relation Matrix
## 5 conclusion
