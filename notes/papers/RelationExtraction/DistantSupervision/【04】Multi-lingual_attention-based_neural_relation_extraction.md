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
从单语言数据集中抽取关系事实，而实际上，人们使用不同的语言对有关真实世界的知识进行描述；并且由于人类经验和人类认知系统的相似性，说着不同语言的人共享着有关真实世界的相似知识**。

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
**key motivation of MNRE**: for each relational fact, the relation patterns in sentences of different languages should be substantially consistent,and MNRE can utilize the pattern consistency and complementarity among languages to achieve better results for RE.

给定一对实体，他们在m种不同语言中对应的句子被定义为
$$T={S_{1},S_{2},\cdots ,S_{m}}$$
其中，$S_{j}$代表j语言中的包含$n_{j}$个句子的句子集合，
$$S_{j}={x_{j}^{1},x_{j}^{2},\cdots ,x_{j}^{n_{j}}}$$
我们的模型针对每个关系r给出一个分数f(T,r),MNRE框架主要包含2个部分：
> 1. sentence encoder:给定一个句子x和2个目标实体，使用CNN 将句子x中的关系模式 编码为一个分布式表示X。
> 2. multi-lingual attention: 当各种语言中的所有句子被编码为分布式表示后，聚集 注意力加权后的所有句子向量，得到全局向量，用于关系预测。
### 3.1 sentence encoder
句子编码器旨在 通过CNN将一个句子x转换为它的分布式表征X。具体地，① 将输入句子中的word 嵌入为一个稠密实值向量，即将word embedding 和 position embedding拼接；② 使用卷积操作(使用卷积层抽取局部特征)、最大池化和非线性变换tanh function(合并所有的局部特征，形成一个全局表示), 来构建句子的分布式表征。

过程与【03】中的句子编码过程类似。经过上述过程后的句子表示向量 被认为能够有效地从输入句子中 编码目标实体对的关系模式。**how the relation patterns is encoded**?
### 3.2 multi-lingual attention
+ mono-lingual attention to select informative sentences within one language

adopt different mono-lingual attentions to de-emphasize noisy sentences within each language. 具体来说，对于j语言中的句子集合Sj来说，
我们**旨在 聚集该语言中所有的句子向量，生成一个实值向量Sj，用于关系预测**。

单语言句子集合向量Sj计算公式如下：
$$\mathbf{S}\_{j}=\sum_{i}\alpha \_{j}^{i}X_{j}^{i}$$
$\alpha \_{j}^{i}$是每个句子向量的attention score，计算方法如下：
$$\alpha \_{j}^{i}=\frac{exp(e_{j}^{i})}{\sum \_{k}exp(e_{j}^{k})}$$
$e_{j}^{i}$是一个基于查询的函数，其得分可以表示 输入句子向量表示$x_{j}^{i}$ 反应其标注关系r 的程度。$e_{j}^{i}$计算方法如下：
$$e_{j}^{i}=X_{j}^{i}\cdot r_{j}$$
上述过程其实就是【03】的 selective attention over instances。
+ cross-lingual attention to measure the pattern consistency among languages

**key idea**:emphysize those sentences which have strong consistency among different languages.具体地，考虑 不同语言中关系模式的对应一致性因素，进一步去掉不太可能的句子，得到更多的集中的、富含有用信息的句子。**跨语言注意力机制 与 单语言注意力机制的工作机制类似**。

跨语言句子集合向量Sjk计算公式如下(j、k分别代表两种不同的语言)：
$$\mathbf{S}\_{jk}=\sum_{i}\alpha \_{jk}^{i}X_{j}^{i}$$
$\alpha \_{jk}^{i}$是j语言中每个句子向量$X_{j}^{i}$ 对应于k语言的cross-lingua attention score，计算方法如下：
$$\alpha \_{jk}^{i}=\frac{exp(e_{jk}^{i})}{\sum \_{k}exp(e_{jk}^{k})}$$
$e_{j}^{i}$是一个基于查询的函数，其得分可以表示 输入句子向量表示$x_{j}^{i}$ 反应其标注关系r 的程度。$e_{j}^{i}$计算方法如下：
$$e_{jk}^{i}=X_{j}^{i}\cdot r_{k}$$
### 3.3 prediction
假设有m种语言，使用 cross-lingual attention，我们得到m×m个实值向量Sjk；将所有的向量Sjk放在一起，定义整体打分函数f(T,r)如下：
$$f(T,r)=\sum_{j,k\in {1,2,\cdots ,m}}logp(r|\mathbf{S}\_{jk},\theta )$$
其中，$p(r|\mathbf{S}\_{jk},\theta )$代表 在给定句子集合表示Sjk 和 参数 $\theta$的情况下，预测关系为r的概率，其使用softmax layer得到：
$$p(r|\mathbf{S}\_{jk},\theta )=softmax(\mathbf{M}\mathbf{S}\_{jk}+\mathbf{d}),\mathbf{M}\in \mathbb{R}^{n_{r}\times R^{c}}$$
其中，矩阵M是 随机初始化的全局关系矩阵。

**为更好地考虑 每种语言的特性，我们进一步引进$\mathbf{R}\_{k}$作为k语言的特定关系矩阵**。因此上述概率计算公式可改写为：
$$p(r|\mathbf{S}\_{jk},\theta )=softmax((\mathbf{R}\_{k}+\mathbf{M})\mathbf{S}\_{jk}+\mathbf{d})$$
其中，矩阵 M 编码global patterns，用于关系预测；矩阵 Rk 编码 language-specific characteristics。

**需要注意的是**：在训练阶段，我们使用标注好的关系来构建 句子集合向量Sjk；在测试阶段，我们为每一个可能的关系r构建不同的句子集合向量Sjk，来计算f(T,r),用于关系预测。
### 3.4 optimization
loss function：
$$J(\theta )=-\sum_{i=1}^{s}f(T_{i},r_{i})$$
此处，s代表 不同语言中的 对应于每一个句子集合的 所有实体对的数量，$\theta $代表模型参数。
## 4 experiments
### 4.1 datasets and evaluation metrics
### 4.2 experimental settings
### 4.3 effectiveness of consistency
### 4.4 effectiveness of complementarity
### 4.5 comparison of Relation Matrix
## 5 conclusion
