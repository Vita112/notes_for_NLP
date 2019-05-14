**基于远程监督的无标注数据的关系抽取** Mike Mintz Steven Bills 2009 斯坦福大学<br>

提出一种适用于各种大小的、未标注的语料库的研究范式-distant supervision，同时避免domain dependence。实验使用一个拥有上千个关系的大型语义数据库Freebase，来实现远程监督。
**算法对于出现在某些freebase relation中的每一对实体，在一个大型的未标注语料库中，找到包含这些实体的所有句子，并抽取文本特征，训练一个关系分类器**，结合了监督IE和无监督IE技术的优点。本文分析了特征性能，显示 对于有歧义，或词汇距离远的句子的关系抽取任务，句法剖析特征特别有用。
## 1 introduction
对有监督学习方法，无监督学习方法，以及半监督bootstrapping方法的概述，此处略。

远程监督是Snow等人所用范式的延伸，该范式利用WordNet提取实体之间的超同名（IS-A）关系，类似于生物信息学中弱标记数据的使用。

**intuition of distant supervison**：如果一对实体参与到freeebaseKB 的某种已知关系，那么，对于未标注的语料库，包含有这2个实体的任何句子都是表达这种关系的 一个实例。以下是**算法优点**：
```
1. 使用更大规模的数据：more text more relations more instances，使用120万wikipedia articles和连接了94万实体的102种关系的180万个实例。
2. do not suffer from 过拟合和域依赖的问题，因为算法使用database进行监督学习
3. 分类器的输出使用 规范的关系名称
4. 整合多个句子的数据，来决定2实体间是否存在一个关系：聚集来自 包含某个实体对的不同句子 的特征，为分类器提供更多的信息，产生更准确的标签
```
**缺点**：某些句子可能并没有表达这样的关系，但句子特征却被加入到
## 2 previous work
+ use little or no syntactic info
>+ DIPRE: string-based regex
>+ SnowBall:learned similar regular expression patterns over words and named entity tags

+ use deeper syntactic info which got from parse of the input sentence

+ freebase:一个免费可用的 在线的 结构化语义数据库，Freebase中的数据主要来自text boxes 和其他来自wikipedia的表格数据。

关系relation:an ordered, binary relation between entities.<br>
关系实例relation instances:individual ordered pairs in this relation<br>
实验中，我们有900万实体间的包含7300中关系的1.16亿个实例，过滤掉那些nameless和uninteresting实体，结果我们得到94万实体间的102种关系的180万个实例。
## 3 model architecture
**assumption**：如果2个实体参与某个关系，那么，任何包含这2个实体的句子可能表达了这样一种关系。算法训练一个多类逻辑回归分类器multiclass logistic regression classifier，为每个噪声特征noisy feature学习权重。
>使用freebase知识库对应到大量的未标注数据中进行自动标注，生成已标注训练数据集(正项)，训练集中的关系和实体对来源于freebase知识库，再加入一些负项，训练一个分类器，最后每一个分类是一个关系。
```
training step:
使用一个NE tagger来定义句子中的所有实体。如果一个句子包含2个实体对，且这2个实体是freebase知识库中的关系的一个实例，那么，就从句子中抽取特征，并将特征添加到关系的feature vector中；
testing step：
使用NE tager定义实体，在一个句子中一起出现的每一对实体，都被认为是一个潜在的关系实例，并从句子中抽取特征，将特征添加到那个实体对的特征向量中。
对测试语料库中每一个句子中的每一对实体，都进行特征抽取，regeression classifier基于来自出现该实体对的所有句子的特征，为每一个实体对预测一个关系名称。
```
+ 模型框架的主要优势是：**能够结合 同一关系的 来自不同mentions的信息**。以下这个句子说明了这种优势：
![examples_for_distant_supervision@2009]()

上面第一个句子可表达：film-director，film-writer，或者film-producer这几种关系，但加上第2个句子的信息，我们可以确定其关系为film-director。
## 4 features
### 4.1 lexical features
describe specific words between and surrounding the two entities.每个词汇特征由以下成分拼接而成：
```
1. 2个实体间的词序列
2. 这些词的 词性标注
3. 一个flag：表明哪个实体在前
4. 实体1的左边的k个单词大小的窗口，以及词的词性标注
5. 实体2的右边的k个单词大小的窗口，以及词的词性标注
```
### 4.2 syntactic features
使用 broad-coverage dependency parser MINIPAR 解析每个句子，来得到句法特征。对于每一个句子，我们抽取出 每一对实体间的一个dependency path。依存路径由一系列表示解析遍历的 依存项、方向、词/块 组成。词性标注不包含在依存路径中。
句法特征由以下项拼接而成：
```
1. 2个实体间的依存路径
2. 对一个实体来说，一个窗口节点不是依存路径的一部分，一个窗口节点window node是一个连接2个实体中的某一个实体的节点。
```
![lexical_features_and_syntactic_features]()

上图table 3中，每一个lexical row代表一个独立的词汇特征；每一个syntactic row代表一个独立的句法特征。
### 4.3 NE tag features
使用Stanford 4-class named entity tagger来进行实体标注，标签器将为句子中每一个单子给予一个标签，共包括person，location，organization，miscellaneous杂项，和none。
### 4.4 feature conjunction
每一个特征由 句子的好几个属性 拼接而成，加上实体标签。要匹配2个特征，要求他们所有的拼接项都正确匹配。这`要求高准确率 和 低召回率`。
## 5 implementation
+ text

使用Freebase Wikipedia Extraction，它是所有维基百科文章的全文的转储，这些文章都是由Freebase的开发人员MetaWeb Technologies标记的句子。

使用wikipedia。因为相对很新；句子倾向于明确很多 新闻报道会忽略的 事实。
+ parsing and chunking

使用MINIPAR对非结构化文本中的每一个句子进行依存解析，产生一个dependency graph。过程中，拥有相同的命名实体标签的 连续的单词 are chunked。
+ training and testing

2种模型评估：held-out evaluation 和 human evaluation。对以上两种实验，我们`只抽取没有出现在训练集中的关系实例,即freebase中没有的实例`.
为构建分类器，系统需要**negative训练数据**，因此，在训练阶段，通过随机选择没有出现在freebase关系中的实体对，抽取其特征，为`unrelated relation`构建一个特征向量。我们从这类实体对中随机取样1%作为负训练样本。
> 我们使用一个带高斯正则化的基于L-BFGS优化的多类logistic分类器。我们的分类器以实体对和特征向量作为输入，并基于属于该关系的实体对的概率，返回关系名和置信度得分。
 ## 5 evaluation
 使用两种方法评估标签：automatically ；manually
 + held-out evaluation
在训练阶段，留出一部分Freebase relation data，在留出的这些数据上对比新发现的关系实例。the combination of syntactic and lexical features offers a substantial improvement in precision over either of these feature sets on its own.
![held-out evaluation](https://github.com/Vita112/notes_for_NLP/blob/74d821bf99a4cafc08cb80ce937c572aebc92088/notes/papers/RelationExtraction/DistantSupervision/pictures/held-out_evaluation.jpg)
+ human evaluation

通过人工来观察每一个被标记为positive label 的实体对，并标记出这个实体对之间是否真的存在一个关系。具体的，由Amazon's Mechanical Turk service的评估人员来进行人工评估。
![human evaluation](https://github.com/Vita112/notes_for_NLP/blob/74d821bf99a4cafc08cb80ce937c572aebc92088/notes/papers/RelationExtraction/DistantSupervision/pictures/human-evaluation_experiments.jpg)

上图显示：在100个实例的情况下，结合词汇和句法特征得到了较高的recall；然而，在1000个实例的情况下，结合两者却没有得到较高的recall。
## 6 discussion
实验结果表明，该远程监控算法 能够为相当多的关系提取高精度的模式。句法特征在远程监督信息抽取任务中确实有效。

在将来，将研究 更简单的、基于块的句法特征是否能够在不需要完全解析的情况下，捕获到 更好的结果，以及共指消解是否有助于提高性能。
