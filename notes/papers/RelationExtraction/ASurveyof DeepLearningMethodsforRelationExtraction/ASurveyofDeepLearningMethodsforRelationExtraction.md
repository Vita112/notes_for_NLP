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
