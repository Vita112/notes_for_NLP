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
>+ matching function:当一些重要的特征在$t_i$,$t_j$见共享时，返回1；否则返回0；
>+ similarity function：当在$t_i$,$t_j$间返回一个正值相似得分。

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
+ roth&yih:first attempt

提出一个 实现识别实体和关系的局部独立分类器 的框架， 通过 一个二分有向无环图的贝叶斯信度网络，来编码实体和关系间的依存关系。在二部图中，实体被作为nodes在一个layer中，关系则作为nodes被表征在其他layer中。每一个关系实例节点$R_ij$从它的参数实体实例节点Ei,Ej有两个传入的边。给定描述句子的特征向量X，局部实体和关系分类器被用于分别计算Pr（Ei|X）和Pr（Rij|X），通过条件概率Pr(Rij|jEi，Ej )编码限制，限制可以从实体和关系标注语料库中被设置和预估。通过最大化贝叶斯网络中nodes的联合概率，获得实体和关系nodes的最可能的标签分配。作者使用2个具体关系进行试验发现：使用贝叶斯网络的关系分类的表现 比 独立关系分类器的 好。**这个结果并不适用于实体分类**。
+ yu&lam：framework based on undirected discriminative probabilistic graphical model。
没有对 实体提及边界的知识 进行假设，而是把它作为 模型的一部分。
+ Singh：models co-refrences jointly with entity mentions and relations

proposed a single，joint undirected graphical model，表征实体抽取、关系抽取和共指消解这三种任务的 各种依存。模型捕获文档中所有实体提及和他们之间的关系以及共指，通过扩展信度传播算法，在推理过程中稀疏变量域，作者解决了 模型中变量过多的 问题。
### 3.3 card-pyramid parsing-joint extraction
+ Kate and Mooney：to jointly label the nodes in the card-pyramid graph. They propose a parsing algorithm analogous to the bottom-up CYK parsing algorithm for Context Free Grammar (CFG) parsing.新的解析算法要求的语法 被称为 card-pyramid grammar，包含以下生成类型：
> Entity Productions of the form EntityType→Entity,e.g.per→leaders<br>
Relation Productions of the form RelationType → EntityType1 EntityType2，e.g.PHYS → PER GPE

示例句子的卡片-金字塔图如下：

![Card-pyramid graph for our example sentence](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/Card-pyramid_graph_for_our_example_sentence.jpg)

### 3.4 structured prediction
+ Li&Ji:提出了一种增量式联合框架an incremental joint framework，用于同时提取实体注释和关系，同时结合了实体注释的边界检测问题

尽管使用了optimal  global decision，训练期间实体抽取和关系抽取间的交互问题仍被禁止。因此，作者建议将该问题重新表述为结构化预测问题。他们试图为一个给定的句子x∈X，预测一个输出y∈Y。线性模型为： 
>${y}'=arg max f(x,y)\cdot \vec{w}$

f(x,y)为描述整个句子结构的特诊向量。他们应用`beamsearch`来逐步扩展输入句子的部分构造，以找到得分最高的结构。该框架的**主要优势**在于：可以很容易的利用这2个任务的任意特征。一些用于实体提取的全局特征 试图捕获实体提及之间的长距离依赖关系，比如：
```
co-reference consistency:在同一个句子中，使用一些简单的启发式规则确定两个段之间的共同引用链接.
neighbour coherence:两个相邻段的实体类型作为一个全局特征链接在一起.
part-of-whole consitency:如果一个实体提及 在语义上是另一个提及的一部分，那么他们应该属于同一个实体类别。
```
一些用于RE的全局特征有：
```
triangle constraint:多个实体的提及不太可能完全与相同的关系类型连接。一个负特征用于惩罚包含此类结构的任何结构.
inter-dependent compatibility:如果2个实体注释由一个依存链连接，他们倾向于有 与其他实体兼容 的关系。
```
+ Miwa and Sasaki：a table structure represent entity and relation structures in a sentence。

句子中单词数量为n，table是一个n\*n的 下三角矩阵lower triangular matrix。various 局部和全局特征 are captured to assign a score to any labels assignment in a table。
### 3.5 evaluation
大多数联合抽取实体和关系的方法，都比基本的`pipeline`方法表现好。联合模型灵活使用 关系信息来抽取实体。但由于没有一个单一的、标准的数据集来反应结果，各种联合模型的方法很难比较。

## 4 semi-supervised approaches
major motivation：减少创建标注数据所需要的人工劳动；利用不需要投入大量精力的，无需标注的数据。
### 4.1 bootstrapping approaches
bootstrapping过程可形象化描述为：`对于给定的NLP任务，选取特定的有指导的，训练分类模型的方法`。要求2个数据集：一个多量的未标注语料库U；一个少量的有标注的关系类型的种子实例L，然后通过未标注数据集来逐步扩大标注数据集，从而训练出最后的分类器实现具体的NLP任务。是一个`使用少量的标注语料，获取到置信度较高的 多量的标注语料的 反复迭代过程`。大致过程如下：
```
1 使用已标注的数据集，应用选择的方法训练分类器h，h用于标注 未标记数据集中的标记分类，可能是一些启发式规则；
2 使用h对U进行标注分类，从U中获取标注数据；
3 从step2中获取的标注数据中，选择置信度较高的数据作为标注数据 ，加入到标注数据集中；
4 重复上述过程，知道满足迭代结束的条件。
```
+ 第一个bootstrapping算法是DIOER(dual iterative pattern relation expansion),由Brin提出。

该算法背后的直觉intuition是：pattern relation duality模式关系对偶。下图是DIPRE对偶迭代模式关系扩展的overview：
![overview_of_DIPRE](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/overview_of_DIPRE.png)

两个实体E1，E2间的 用于捕获关系类型R的 模式由一个5元素元组表征：(order,urlprefix,prefix,middle,suffix),其中，order是一个布尔值，其他都为字符串。
例子：
> a pattern is (true, \en.wikipedia.org/wiki/", City of, is capital of, state) and it matches a text like City of Mumbai is capital of Maharashtra state.
+ Agichtein and Gravano:基于DIPRE发展了一个叫`snowball`的系统。

有2点优于dipre：
>+ ①模型表征和泛化;②模式和元组的评估。
>+ 关于①，snowball的关键先进点之一是：在模式中包含命名实体标记（PER、ORG、LOC等）。在dipre模式中，要求prefix、suffix和middle strings完全匹配。这阻碍了模式的覆盖范围。在SNOWBALL中，文本中的细微变化，比如拼写错误和附加文章，不会导致错误匹配。在向量空间模型中使用词向量woed vector，2个上下文词向量间的点积越高，相似度越高。
>+ 关于②，snowball丢弃了所有不够精确的patterns。一个方法是：过滤掉一些 最小数量的种子示例不支持的所有模式。snowball基于 认为2个NE中的1个比另一个更重要 的假设，为每个pattern计算置信度confidence，p的置信度被定义为：
![confidence_defined_in_snowball](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/a%20survey%20on%20RE/pictures/confidence_defined_in_snowball.png),
此处，#positive_p and #negative_p are the numbers of positive and negative matches for p,respectively.由于每一个词迭代都丢弃了 低置信度的模式和元组，从而避免了很多不正确的抽取。

+ Gabbard et al. :explore the use of co-reference information

+ Mention level- Zhang:提出一种基于SVM的自举引导算法bootProject.通过放宽对多个特征“视图”的以下限制来 推广联合训练算法：互斥、条件独立性和分类充分性.

+  Sun :提出一个`二阶段`自举方法a two-stage bootstrapping approach

观察到`dipre`和`snowball`在抽取general relations like EMP-ORG relation in ACE时，表现并不是很好。他们提出一个`二阶段`自举方法, the first stage is similar to SnowBall,the second stage将第一阶段学到的模式作为inputs,试图抽取更多nominals ,like manager, CEO, etc.基于这些学习的名词列表的特征 被合并到受监督的RE系统中.此外，一个多量的未标注语料库被用于学习`word clustering`，因此，出现在相似上下文中的words被分组到相同的cluster中。

+ **有一点需要注意**：基于bootstrapping的算法的表现依赖于`the choice of initial seed examples`。关于算法中种子样本的选择的分析参见` Vyas et al., Kozareva and Hovy`的论文。
### 4.2 active learning
active learning 算法背后的关键思想是：允许学习算法查询某些选定未标记实例的真实标签。主动式学习通过很少的标注实例可以获得与监督学习方法相当的性能。
+ Sun and Grishman：LGCo-Testing

为应用Co-Testing，它们提出`创建关系实例的两个视图view`。
>a local view based on features，这些特征捕获 被连接的实例mentions和包含句子containning sentence的其他特征。<br>
>a global view based on 连接2个实例mentions的短语phrase的分布相似性，使用一个大型语料库。<br>
>在两个相似的phrase间，分布相似性将为这2个phrase分配很高的相似性，如果这些phrases被观察到出现在一个大型语料库中的相似上下文中。使用local view的feature可以训练`a maximum entropy classifier`。当一个分类器使用global view，使用分布相似性 寻找最近邻居的 一个最近邻分类器被使用。
### 4.3 label propagation method
+ Zhu and Ghahramani：graph based semi-supervised method

数据中的标注和未标注实例被表征为一个graph中的带有edges的节点，edges反应了节点间的相似性。方法中，任何节点的标签信息通过加权edges迭代地被传递给临近的节点，最后，当传递过程收敛时，未标注样本的标签被推断出来。
+ Chen et al：第一个将label propagation method 应用于RE。将特征向量作为特征上的概率分布，利用JS散度计算任意两个关系实例之间的距离。2个实例间的相似性则与这个距离成反比。

+ label propagation的一个主要优点：未标注实例的标签不仅由临近的标注实例决定，还由邻近的未标注实例决定。

### 4.4 other methods
### 4.5 evaluation
相比于捕获实体对的每一次提及，这些bootstrap based 技术创建了 一个展示一个特殊关系类型的实体提及对的 列表。通过验证所有抽取出的**对**
，我们很容易测量precision，但是评估recall却很难。因此，需要考虑一个更小的未标注数据的子集。
## 5 unsupervised relation extraction
### 5.1 clustering based approaches
+ Hasegawa et al：propesed one of the earliest approaches,only require a NER tagger to identify named entities in the text.
方法的步骤如下：
```
1 标注文本语料库中的NE;
2 生成共现co-occurring NE，记录他们的上下文；
3 计算step2中定义的所有NE对间的上下文相似性；
4 使用上一步中计算的相似值，聚类NE对；
5 因为每一个聚类代表一个relation，一个标签被自动分配给每一个聚类，来描述由他所表征的关系类型
```
解释几个概念：
>+ Named Entity pairs and context
>+ context similarity computations
>+ clustering and labelling
### 5.2 other approaches
## 6 open information extraction
+  Banko et al. :TextRunner consists of following 3 core modules
>1. self-supervised learner
>2. single pass extractor
>3. redundancy-based assessor

proposed to use CRF based, self-supervised sequence classifier O-CRF instead of Naive Bayes classifier used in
TextRunner and observed better performance.
+ Fader et al.:ReVerb
TextRunner有以下限制：
>1. incoherent extracitons
>2. uninformative extraction
>3. overly-specific extractions
为克服以上限制，ReVerb算法对要提取的关系短语提出以下两个约束：
>syntactic constraint
>lexical constraint
## 7 distant supervision:combines advantages of both the paradigms : Supervised and Unsupervised
+ Mintz et al. :proposed Distant Supervision, used Freebase as a semantic database which stores pairs of entities for various relations.

>1. labelling heuristic
>2. negative instances

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
