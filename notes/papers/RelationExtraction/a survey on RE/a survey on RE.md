## abstract
随着互联网信息的快速发展，每天有海量的数字文本信息生成，包括论文，公开研究成果，博客，问答论坛等。自动抽取隐藏在这些文本信息背后的**知识**，将提高个人以及企业的工作效率。**RE**的任务是 _identify these relations between entities automatically._ 本文将介绍关系抽取任务中一些重要的方法：supervised,semi-supervised,unsupervised RE techniques,and open information extraction,distance supervision.
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
+ Zhou et al. 《 Biomedical relation extraction:From binary to complex》surveys most of the recent biomedicalRE approaches

## 2. supervised approaches
关注mention level的关系抽取，要求标记数据，其中每对实体引用都标记有一种预定义关系类型(包含NONE)。通常被阐述为一个`muti-class classification problem`，每一类都对应一个不同的关系类型。基本假设是：训练数据集和测试数据集来自相同的分布。
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
rK(s; t) = fbK(s; t) + bK(s; t) + baK(s; t) + mK(s; t).<br>
>fbK:fore-between subkernel, 计数fore-between patterns,例如(president PER of ORG).<br>
bK:between subkernel，计数between patterns,例如(PER joined ORG).<br>
baK:between after subkernel, 计数between after patterns，例如(PER chairman of ORG announced).<br>
mK：modifier subkernel, 当2个实体标注间没有其他单词，且第一个标注作为修饰器时，计数 $x_{1}x_{2},y_{1}y_{2}$ 间的共享子序列数量。

使用基于这种关系kernel(rK)的SVM分类器。

#### 2.2.2 syntactic tree kernel

句子的结构属性通常由它的成分解析树编码，受context free grammar(CFG)支配。
+ collins&duffy：convolution parse tree kernel计算任意2个句法树之间的相似性，计算两个句法解析树之间共享的子树。

在一转换空间中，一个句法树T的映像是：$h(T)=\left \[ h_{1}(T),h_{2}(T),\cdots ,h_{n}(T)\right ]$.其中$h_{i}(T)$代表在句法树T中第i个子树出现的次数，n代表可能子树的数量。计算公式见原文。
+ zhang et al.:描述了5种情形来为一个给定关系实例构建一个树表征。

>最小完全树(MCT):一个由2个实体的最小共同祖先构成的完全子树；<br>
path-enclosed tree(PT):包含尸体的最小的subtree。<br>
context-sensitive path tree(CPT)：PT的扩展版，包含第一个实体的左边一个词和第二个实体的右边一个词。<br>
flattened path-enclosed tree(FPT)：pt修改版，只有单个in 和 out 弧线的 non-pos non-terminal nodes 会被忽略。<br>
flattended context-sensitive path tree(FCPT):cpt修改版，只有单个in 和 out 弧线的 non-pos non-terminal nodes 会被忽略.

其中，当用来计算Kt时，pt是表现最好的。
+ zhou et al.: 为RE自动决定一个动态的上下文敏感树跨度。

通过扩展pt，提出了上下文敏感的卷积树核，该核除了考虑上下文无关的子树外，还考虑上下文敏感的子树作为它们的上下文。
+ qian：使用constituent dependencies的信息来动态决定树跨度。

+ sun & han：feature-enriched tree kernel，在句法树中，使用一套判别特征为nodes注释。

总结图如下：

![syntactic_parse_tree_kernel](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/syntactic_parse_tree_kernel.png)

#### 2.2.3 dependency tree kernel

+ culotta & sorensen：提出一个kernel来计算2个依存树之间的相似性，是为shallow parse tree representation浅解析树表示定义的tree kernel的扩展。

对于在句子中的每一对实体标注，它们考虑到了包含标注的 句子依存树的最小子树。在依存树中每个节点通过传统特征，如pos tag，chunk tag等，得到加强。形式上，一个关系实例通过拥有节点{t0……tn}的加强型依存树表示，每一个node $t_i$拥有特征Φ($t_i$)={v1……vd}.下图是一个例句的依存树，在右边图的情况下，t0\[c]=t\[{0,1}]={$t_1$,$t_2$},且$t_1$.p=$t_0$.为比较任何两个nodes$t_i$,$t_j$,定义了以下2个函数：<br>

![dependency tree](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/dependency_tree.png)
>matching function:当一些重要的特征在$t_i$,$t_j$见共享时，返回1；否则返回0；similarity function：当在$t_i$,$t_j$间返回一个正值相似得分。
+ 当一对matching nodes被找到时，它们的孩子的所有可能matching的子序列也会被找到。在这样的matching 子序列中所有节点的相似得分总和相加后，得到孩子节点的整个相似性。

+ harabagiu et al.:使用propbank 和 framenet 利用从浅语义解析器中得到的语义信息，来加强这个依存树。

#### 2.2.4 dependency graph path kernel
+ Bunescu and Mooney：提出一种基于kernel的新的依存路径。

直观来说，使用依存图2个实体间的最短路径，可以获得 明确断言一个句子中2个实体间的关系的 所要求的信息。kernel被设计用于捕获表示2个关系实例的 最短依存路径间的 相似性。由于完全的词汇化路径可能导致数据稀疏，因此，对单词进行了不同程度的泛化，将单词类别化为单词类，比如pos tag，泛化的pos tag等。缩短依赖路径核计算2个关系实例间共享的常用路径特征的数量。`这种依存路径核强化了这种限制：2个路径应该由完全相同的nodes`。

#### 2.2.5 composite kernels
一个复合的kernel可以把有单个kernel捕获到的信息结合起来。例如，结合由tree kernels捕获的句法信息 和 由sequence kernel捕获到的词汇信息。kernel结合的有效方法有：sum，product，linear combination。
+ zhang：结合句法树kernel 和 实体kernel，构建了2个复合kernel。
+ zhao&grishman：结合来自nlp过程中的3个不同层次的信息，这样出现在一个层次的processing error 可以被 其他层次的信息 解决。
+ wang：使用a sub-kernel defined using relation topics。

### 2.3 evaluation
如下图：

![evaluation_for_supervised_methods](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/evaluation_for_supervised_methods.png)

从precision，recall，F-measure of non-NONE classes 三方面评估。尽管通过不同的方式使用了相同的数据，但是，5次交叉验证中使用的实际拆分/折叠可能不同。我们得知：基于kernel的方法 比基于feature的方法 表现更优秀；其中，基于句法树kernel方法表现最好。

## 3 joint extraction of entities and relations
上节中介绍的方法都基于这样一种假设： the knowledge about boundaries and types of entity mentions are known before hand.先定义实体mention和实体类型，然后在使用RE技术。这种‘pipeline’方法容易出现传递错误propagation errors from extracting entity mentions phase to extracting relations phase。以下方法的目的是避免这种传递错误的出现。
### 3.1 integer linear programminig based approach
+ roth&yih：提出一个模型，首先为实体抽取学习独立的局部分类器，然后RE。

在推论过程中，有一个限制:如果接受局部分类器的建议，将违背之前所讲的域限制。为克服这种限制，提出了整数线性规划：它最小化 assignment cost function 和 constraint cost function的和。在使用了ILP做全局推论时，表现很好。
+ chan&roth：扩展原始的ILP框架，结合背景知识，比如关系类型的层次，共指信息等。
### 3.2 graphical medels based approach
### 3.3 card-pyramid parsing
### 3.4 structured prediction

## 4 semi-supervised approaches
### 4.1 bootstrapping approaches
### 4.2 active learning
### 4.3 label propagation method
### 4.4 other methods
### 4.5 evaluation
## 5 unsupervised relation extraction
### 5.1 clustering based approaches
### 5.2 other approaches
## 6 open information extraction
## 7 distant supervision
## 8 recent advances in RE
+ universal schemas by riedel：使用通用范式，即现有结构化数据库的关系类型的联合，以及OPEN IE中使用的表面形式的所有可能关系类型。

提出一种方法，从这些通用关系类型中学习不对成话语含义asymmetric implicature。但由于含义的不对称性，导致不可反推。
+ n-ary relation extraction：多于2个以上的实体间的关系通常被认为是 `复杂，高阶或n元关系`。

McDonald et al. ：used well-studied binary RE to initially find relations between all possible entity pairs.`THEN`, find 最大的圈子maximal cliques in this graph such that each clique corresponds to some valid n-ary relation.在biomedical domain 数据集上演示了这种方法的效果。*另一个视角是：将n-ary RE 问题看作是一个 语义角色标注的e问题*。
+ Cross-sentence Relation Extraction句际关系抽取:Swampillai and Stevenson   **感兴趣，可看相关论文**

proposed an approach to extract both intra-sentential and inter-sentential relations。作者在处理句际关系问题时，使用了句内关系所用的结构化特征，比如parse tree paths，和技术。通过`co-reference resolution`可以解决大多数问题。
+ Convolutional Deep Neural Network：

zeng：Relation classification via convolutional deep neural network。employed a convolutional DNN to extract lexical and sentence level feature。
+ Cross-lingual Annotation Projection跨语言注释投影    **另一个感兴趣方向**

A cross-lingual annotation projection approach for relation detection by Kim，使用平行语料库，实现从源丰富的源语言到源稀少的目标语言的关系注释。Kim&Lee提出一种基于图的投影方法，该方法利用由实体和上下文信息共同构建一个graph，并以交互方式操作。
+ Domain Adaptation：

当训练数据集和测试数据集来自不同的分布，监督系统被用于分类out-of-domain数据时，监督方法出现降级。 Plank and Moschitti指出，通过将词聚类和潜在语义分析（LSA）得到的语义相似信息嵌入到句法树核中，可以提高基于核系统的域外性能。
## 9 conclusion and future research directions
