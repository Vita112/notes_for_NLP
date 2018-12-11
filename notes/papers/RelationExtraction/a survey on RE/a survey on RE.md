## abstract
随着互联网信息的快速发展，每天有海量的数字文本信息生成，包括论文，公开研究成果，博客，问答论坛等。自动抽取隐藏在这些文本信息背后的**知识**，将提高个人以及企业的工作效率。**RE**的任务是 _identify these relations between entities automatically._ 本文将介绍关系抽取任务中，重要的supervised，semi-supervised and unsupervised RE techniques。以及 open information extraction and distance supervision.
## 1. Introduction
IE的主要目标是从给定文本库中抽取某种特定信息，输出一个结构性知识库，比如一个关系表，或者XML文件。通常用户想从文本中抽取的信息有三种：
+ named entities
+ relations 
+ events

一个实体（NE）通常是一个`代表真实世界里特定物体的` 词语或短语.generic NE types有：person，organization，location，date，time，
phone，zipcode，email，url，amount etc.以及film-title，book-title等，在`fine-gained  NER`（精细NER）中，面临的问题是：识别那些分层级的泛型实体。
NER的任务是：identifying all the mentions or occurrences of a particular NE type in the
given document。<br>
一个relation代表一个well-defined(have a specific meaning) relationship between 2 or more NEs.我们```focus on binary relations and
assume that both the argument NE mentions that participate in a relation mention occur in the same sentence.``` 需要注意**并不是每一个实体对之间都存在一个关系**。re需要检测出提及的实体，决定实体见得关系。RE面临的挑战
```存在大量类目繁多的可能关系
   处理非二元关系non-binary relation 面临特殊挑战
   有效数据集的缺乏是RE中使用监督机器学习面临的问题
   inherent ambiguity of what a relation means
   ```
### 1.1 datasets 
ACE：automatic content extraction，一个由NIST主导的评估，用于评价EDT(entity detection and tracking)和RDC(relation detection and characterization).它定义了一下NE类型：PERSON,ORG,LOCATION,FACILITY,GEO_POTITICAL_ENTITY(GPE) etc。
### 1.2 RE：global level VS mention level
Global level：produce a list of entity pairs for which a certain semantic relation exists.
Mention level:将实体对和包含他的句子作为输入，然后识别该实体对中是否存在一个特定的relaition。
### 1.3 previous survey
+ sarawagi  《Information extraction. Foundations and trends in databases》
+ Abreu et al. 《A review on relation extraction with an eye on portuguese》cavers various RE techniques used for portuguese language.
+  Zhou et al. 《 Biomedical relation extraction:From binary to complex》surveys most of the recent biomedicalRE approaches

## 2. supervised approaches
关注mention level的关系抽取，要求标记数据，其中每对实体引用都标记有一种预定义关系类型(包含NONE)。通常被阐述为一个`muti-class classification problem`，每一类都对应一个不同的关系类型。
### 2.1 feature-based methods
对于标注数据中的每一个关系实体，生成一个特征集合，训练出一个分类器，用于分类新的关系实体。
+ Kambhatla : 训练了一个maximum entropy分类器，包含49 classes。

examples described are as follow:
![examples_described_by_Kambhatla](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/examples_described_by_Kambhatla.png)
+ Zhou et al. :增加了一些重要特征。如`word based features`,`base phrase chunking based features`,`features based on semantic resouces`.

features based on semantic resouces：一个例子：一个个人亲属关系触发词列表(trigger word list)被用来区分6种个人社会关系的子类。它从wordnet上收集所有拥有语义类“relative”的单词，形成单词列表，然后被分类为代表每个社会关系子类的类别.一个子例：当在ET2是第二种标记的实体，SC1是第一种语义类标注的trigger list中，的一个标注被找到了，那么特征 SC1ET2就生成了。
> 使用了SVM classifier，并发现：phrase based chunking features contribute the most to the accuracy。一个主要原因是：ACE数据中的大多数都是short-distance 关系，而像单词和快特征这些 简单特征就足以识别这些关系。
+ Jiang and Zhai：a systematic study of the future space for RE.

定义了特征空间的统一图形表示unified graphic representation，实验了三个特征子空间，分别对应于序列sequences，句法分析树syntactic parse tree，依存解析树dependency parse trees。发现：syntactic parse tree最有效；在每个特征子空间中，只使用基本单元特征就可以得到好的表现，而增加更多复杂特征可能不会使表现提高多少。

+ Nguyen et al. ：使用SVM在wikipedia实体中识别关系。半自动化建立一个关键词列表以为每种关系类型提供线索。

提出一个新的概念：core tree，来代表任何关系实体。core tree不仅由依存树中连接2个实体标注的最短路径组成，还包括连接 句子中在最短路径和关键词之间的节点node 的额外路径additional path。core tree的子树会被当作特征 去挖掘。
+ Chan and Roth

发现： all ACE 2004 relation types are expressed in one of several constrained syntactico-semantic structures。
> 一个主要问题：calss imbalance。导致更高的precision和更低的recall，因为分类器更倾向于产出更多的NONE类。
方法优点：一旦特征被设计，可以十分容易的应用ML的任何分类器。但，需要花费很多时间去设计the ‘right’ set of features，需要对每个特征做细致的分析，以及语言现象的知识。

### 2.2 kernel methods
在基于kernel方法中，kernel function被设计用于计算2个关系实体之间的representations的相似度，使用SVM进行分类。
#### 2.2.1 sequence kernel
关系实例被表征为sequences，kernel计算任意2个序列之间的共享序列的数量。
+  Bunescu and Mooney：把第1个标注的词序列考虑到句子中的第2个标注中。生成每个单词的特征向量，每个关系实例被表征为特征向量的序列，一个特征向量对一个单词。特征来自以下domains：

![features domains](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/features_domains_for_kernel_method.jpg)

下图显示了关系实例的特征向量序列，此处每一行是一个特征向量。
![特征向量序列](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/Example_of_sequence_of_feature_vectors.jpg)
![特征向量序列s,t的说明](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/interpretation_for_sequencsS%2CT_of_featrue_vectors.jpg)

作者将RELATION KERNEL定义为4个subkernel的总和，每个描述一种特定的类型，基于生成的subsequence kernel。公式表示为:
rK(s; t) = fbK(s; t) + bK(s; t) + baK(s; t) + mK(s; t).
fbK:fore-between subkernel, 计数fore-between patterns,例如(president PER of ORG).<br>
bK:between subkernel，计数between patterns,例如(PER joined ORG).<br>
baK:between after subkernel, 计数between after patterns，例如(PER chairman of ORG announced).<br>
mK：modifier subkernel, 当2个实体标注间没有其他单词，且第一个标注作为修饰器时，计数 $x_{1}x_{2},y_{1}y_{2}$ 间的共享子序列数量。

