title：neural relation extraction with selective attention over instances(基于多实例选择性注意力模型的神经关系抽取)

清华 Lin et al. 2016
## abstract
+ propose a sentence-level attention-based model 
> **first**, use CNN to embed semantics of sentences; **then**,bulid sentence-level attention over multiple instances.

+ contributions:

make full use of all informative sentences of each entities; 

reduce the influence of wrong labelled instances effectively.
+ souce code: https://github.com/thunlp/NRE.
## 1 introduction and related work
KBs: mostly compose of relational facts with triple format, like(Microsoft, founder, Bill Gates); incomplete to contain infinite real-world facts.
> to enrich KBs, many efforts have been invested in automatically finding unknown relational facts.**RE**: the process of generating
relational data from plain text.
### 1.1 RE with distant supervision
+ **Mintz et al. 2009** : 《distant supercision for relation extraction without labeled data》
> propose **distant supervision** to automatically generate training data **via aligning KBs and texts**.     
do not need to manually design features, **but suffer from wrong labelling problem**.
+ **Riedel et al.2010, Hoffmann et al.2011, Surdeanu et al.2012**: adopt multi-instance learning to address wrong labellinig problem.**that is because multi-instance learning consider the reliability of the labels for each instance**.

《modeling relations and their mentions without labeled text》:multi-instance single-label learning

《knowledge-based weak supervision for information extraction of overlapping relations》:multi-instance multi-label learning

《multi-instance multi-label learning for relation extraction》:multi-instance multi-label learning
 
> using NLP tools, like POS tagging, lead to **error propagation**. Because the accururacy of syntactic parsing decrease significantly 
with increasing sentence length.
+ **Socher et al.2012, Zeng et al.2014, dos Santos et al.2015, Zeng et al.2015**: employ deep learning methods

《semantic compositonality through recursive matrix-vector spaces》: use RNN to automatically learn featuers

《relation classification via convolutional deep neural network》:adopt end-to-end CNN

《classifying relations by ranking with convolutional neural networks》:adopt end-to-end CNN

《distant supervison for relation extraction via PCNN》:apply PCNNs to capture structural information between 2 entities;
employ multi-instance learning to address wrong label problem

> base on a sumption:提及这2个实体的句子中，至少有一个句子表达了他们的关系，而且在训练时，针对每一对实体，仅选择那个最可能的句子，来进行预测。**这导致：模型会丢失掉 包含在未观察到的句子中的 大量丰富有用的信息**。

### 1.2 what's new in this paper?

![architecture_of_sentence-level_attention-based_CNN]()

as described in Fig.1, mainly contain 3 steps:
> 1. use CNN to embed the senmatics of sentences
> 2. represent the relation as semantic composition of sentence embeddings（将关系表征为句子嵌入的语义复合）
> 3. build sentence-level attention over multiple instances,and extract relation with the relation vector
weighted by sentence-level attention.

### 1.3 attention model
[Reference blog1](https://blog.csdn.net/malefactor/article/details/78767781)

[Reference blog2](https://blog.csdn.net/mpk_no1/article/details/72862348)
  
[attention_model](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/attention_model.md)

## 2 Methodology

