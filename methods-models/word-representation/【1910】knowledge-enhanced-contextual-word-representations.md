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

**整个KAR被插入在BERT的中间两层之间(inserted between 2 layers in the middle of BERT)**.

## 2 related work
## 3 KnowBert
### 3.1 pre-trained BERT
### 3.2 Knowledge Bases
### 3.3 KAR-knowledge attention and recontextualizztion
### 3.4 training procedure
## 4 experiments
### 4.1 setup
### 4.2 intrinsic evauation
### 4.3 downstream tasks
## 5 conclusion
