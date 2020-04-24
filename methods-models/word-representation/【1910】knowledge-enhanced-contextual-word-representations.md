paper [link](https://arxiv.org/abs/1909.04164)   	EMNLP 2019
## abstract 
* 从unstructured unlabeled text上训练得到的 contextual word representations不包含任何对真实世界实体的explicit grounding，且经常不能remember facts about those entities。

* KBs可以提供丰富的高质量、人类创造的知识源；包含原始文本中的补充信息；编码factual knowledge，这些知识由于很少提及常识性知识或者long range dependencies，很难从选择偏好中学习。

* 提出一种方法：embed multiple knowledge bases into large scale models，use structured，human-curated knowledge enhance word representations。


**the entity linkers and self-supervised language modeling objective are jointly trained end-to-end in a multitask setting that combines a small amount of raw text(在结合了少量原始文本的多任务设置中进行端到端联合训练)**.
## 1 introduction
large scale pretrained model such as ELMo, GPT, and BERT在a wide range of NLP tasks中significantly improved the state of the art.

* insert multiple KBs into a large pretrained model with a Knowledge Attention and Recontextualization(KAR) mechanism.
> for each KB,first, explicitly model *entity spans* in the input text, use an integrated entity linker to retrieve relevant entity embeddings from a KB **to from knowledge enhanced entity-span representations**;

> then,**recontextualized** the entity-span representations with  **word-to-entity atttention** ，来允许contextual word representations 和上下文中所有实体跨度之间的 long range interactions。

**整个KAR被插入在BERT的中间两层之间(inserted between 2 layers in the middle of BERT)；在未标注数据上学习entity linkers with self-supervision**。benefits of our approach are as follow：
> 1. 没有改变the top layers of the original model，因此在训练KAR时，**可以保留输出损失层，并在未标记语料库上进行微调，这允许在任何下游应用中都可以简单地从BERT切换到KnowBert**。
> 2. 利用原始模型的已有的高容量层，KAR是轻量级的，只添加了最少的额外参数和运行时间。
> 3. 吸收其他额外的KBs十分容易，只需要将他们插入到另外的locations。

使用a mix of intrinsic and extrinsic tasks 来评估KnowBert，外部评估显示在关系抽取，实体类型和词义消岐等任务上，任务表现有提升。
## 2 related work
* 2.1 pretrained word representations: *learning context-sensitive embeddings*

notes of reference paper link [1-EMLo-deep-contextualized-word-representations](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/%E3%80%901802-ELMo%E3%80%91deep-contextualized-word-representations.md) ， [2-BERT-pre-training-of-deep-bidirectional-transformers-for-language-understanding](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/%40201905_BERT-pre-training_of_deep_bidirectional_Transformers_for_language_understanding.md)

* 2.2 entity embeddings:*从外部知识源 生成 连续向量表示*

基于知识图谱的方法优化知识图谱中观察到的三元组的得分，通过2个主要的策略：**translational distance models which use a distance-based scoring function**和**linear models which use a similarity-based scoring function**

TuckER是啥？

* 2.3 entity-aware language models

adding KBs to generative LMs：[Reference-Aware-Language-Models](https://arxiv.org/pdf/1611.01628.pdf)
> 模型将 reference视为explicit stochasitic latent variable，该构架允许模型通过访问external databases和internal state来创建实体及其属性的mentions。**这有助于将 可在数据库上或语篇上下文中的可预测位置上能访问的信息合并在一起，即使the targets of the reference are rare words。论文的模型变体是基于确定性注意力的**。

building entity-centric LMs：[Dynamic entity representations in neural language models](https://arxiv.org/pdf/1708.00781.pdf)
> 提出一个语言模型EntityNLM：可以显式地建模实体，动态地更新它们的表示，并在上下文中生成它们的mentions。可以在上下文中对任意数量的实体建模，同时以任意长度生成每个实体。

**这种关注实体的语言模型引入了隐变量，这些变量要求训练全标注，或者边缘化**
* 2.4 task-specific KB architecture
intergrate KBs into neural architecture for specific downstream tasks

[Explicit utilization of general knowledge in machine reading comprehension](https://arxiv.org/pdf/1809.03449)
> **MRC模型和人类的阅读理解之间存在的gap表现在：对大量带有answer spans的passage-question pairs的需求 和 对noise的鲁棒性上。MRC models对noisy data十分敏感，鲁棒性差。**。该论文explore how to intergrate the neural networks of MRC models with the general knowledge: **使用WordNet从每一个给定的passage-question pair中抽取inter-word semantic connections 作为general knowledge**；**提出一个end-to-end MRC named as Knowledge Aided Reader，它显性地使用上面抽取到的general knowledge来帮助模型的注意力机制**。

*KBs中存在着 大量以机构化形式存储的general knowledge。常见的KBs有：①WordNet storing semantic knowledge；② ConceptNet storing commonsense knowledge；③ FreeBase storing factoid knowledge*
> 如何理解显性利用抽取到的 general knowledge来帮助模型的attention mechanisims？


### 3 KAR-knowledge attention and recontextualizztion
### 3.4 training procedure
## 4 experiments
### 4.1 setup
### 4.2 intrinsic evauation
### 4.3 downstream tasks
## 5 conclusion
