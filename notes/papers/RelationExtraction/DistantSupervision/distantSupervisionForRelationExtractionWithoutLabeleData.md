基于远程监督的无标注数据的关系抽取 Mike Mintz Steven Bills 2009 斯坦福大学<br>

提出一种适用于各种大小的、未注语料库的研究范式，distant supervision。结合了监督IE和无监督IE技术的优点。分析了特征性能，显示 对于有歧义，或词汇距离远的句子的关系抽取任务，句法剖析特征特别有用。
## 1 introduction
对有监督学习方法，无监督学习方法，以及半监督bootstrapping方法的概述，此处略。**intuition of distant supervison**：如果一对实体在freeebaseKB中具有某种关系，那么，语料库中，包含有这2个实体的任何句子都是含有这种关系的 一个实例。以下是算法优点：
```
1. 使用更大规模的数据：more text more relations more instances
2. do not suffer from 过拟合和域依赖的问题
3. 分类器的输出使用 规范的关系名称
4. 整合多个句子的数据，来决定2实体间是否存在一个关系：聚集来自 包含某个实体对的不同句子 的特征，为分类器提供更多的信息，产生更准确的标签
```
## 2 previous work
+ use little or no syntactic info
>+ DIPRE: string-based regex
>+ SnowBall:learned similar regular expression patterns over words and named entity tags

+ use deeper syntactic info which got from parse of the input sentence
+ freebase:一个免费可用的 在线的 结构化语义 数据库
relation:an ordered, binary relation between entities.<br>
relation instances:individual ordered pairs in this relation

## 3 model architecture
**assumption**：如果2个实体参与某个关系，那么，任何包含这2个实体的句子可能表达了这样一种关系。算法训练一个多类逻辑回归分类器，为每个噪声特征学习权重。
>使用freebase对应到大量的未标注数据中，生成大规模训练数据集，训练集中的关系和实体对来源于freebase知识库。
```
training step:
如果一个句子包含一个实体对，且这个实体对是freebase知识库中的关系实例之一，那么，会从句子中抽取特征，并将特征添加到
关系的feature vector中
testing step：
在一个句子中一起出现的每一对实体，都被认为是一个潜在的关系实例，并从句子中抽取特征，将特征添加到那个实体对的特征向量中
```
+ 模型框架的主要优势——结合 同一关系的 来自不同mentions的信息
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
使用 broad-coverage dependency parser MINIPAR 解析每个句子。依存路径右一系列表征解析遍历的 依存项、方向、词/块 组成。词性标注不包含在依存路径中。
句法特征由以下项拼接而成：
```
1. 2个实体间的依存路径
2. 对一个实体来说，一个窗口节点不是依存路径的一部分
```
### 4.3 NE tag features
### 4.4 feature conjunction
每一个特征由 句子的好几个属性 拼接而成。要匹配2个特征，要求他们所有的拼接项都正确匹配。这`要求高准确率 和 低召回率`。
## 5 implementation
+ text
使用wikipedia。因为相对很新；句子倾向于明确很多 新闻报道会忽略的 事实。
+ parsing and chunking
拥有相同的命名实体标签的 连续的单词 are chunked
+ training and testing
2种模型评估：held-out evaluation 和 human evaluation。在实验中，我们`只抽取没有出现在训练集中的关系实例,即freebase中没有的实例`.
