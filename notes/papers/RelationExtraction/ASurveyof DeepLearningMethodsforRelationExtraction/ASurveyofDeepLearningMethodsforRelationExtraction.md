比较不同DL模型在RE任务中的表现
## 1 introduction
在监督方法中，关系抽取和分类任务指的是，使用含有实体对mentions的文本，将实体对分类为一组已知关系。RE具体指，预测一个给定文本中的实体对是否包含一个关系。RC具体指，
假定文档包含一个关系，则预测该文档指向的给定本体中的哪个关系类。加上一个额外类NoRelation，这两个task可以结合为一个multi-class 分类问题。
+ 传统监督方法中，RE方法通常由两类:`feature-based method and kernel-based method`。2种方法都需要大量的人工标注语料，且对语料标注精确度要求较高。
