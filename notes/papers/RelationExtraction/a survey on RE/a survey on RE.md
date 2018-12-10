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
assume that both the argument NE mentions that participate in a relation mention occur in the same sentence.``` 需要注意**并不是每一个尸体堆之间都存在一个关系**。re需要检测出提及的实体，决定实体见得关系。RE面临的挑战
```存在大量类目繁多的可能关系
   处理非二元关系non-binary relation 面临特殊挑战
   有效数据集的缺乏是RE中使用监督机器学习面临的问题
   inherent ambiguity of what a relation means```
   
###　1.1 datasets 
