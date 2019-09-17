[paper link](https://arxiv.org/abs/1909.04273)

**Highlights in this paper**:

1. They decompose the joint extraction task into 2 inner-related subtasks:HE(head-entity) extraction and TER(tail-entity and relation) extraction;

2. The proposed 2 subtasks are deconstructed **hierarchically** into several sequence labeling problems based on the span-based tagging scheme;

3. following the **span-based tagging scheme**, They use a general module named HBT(heirarchical boundary tagger) to extract HE and TER, and propose **a multi-span decoding algorithm** to adapt to the multi-target extraction task.

## abstract
prior works for joint extraction of entities and relations **suffer from**:
+ redundant entity pairs may mislead the classifiers;
+ ignoration of important inner structure in the process of extracting entities and relations
## 1 introduction
传统的pipelined methods：忽视了实体抽取和关系预测之间的relevance

feature-baesd joint models：need complicated process of feature engineering

nueral joint models：don't pay much attention to overlapping relations

## 2 methodology
**step 1**:decompose the joint extraction task into 2 inner-related subtasks

+ subtask#1 :HE extraciton- distinguishing all the candidate head-entyties that may be involved with target relations
+ subtask#2:TER extraction- labeling corresponding tail-entities and relations for each extracted head-entity,并且考虑到了 semantic and position information of the given head-entity.

> **one head-entity can interact with multiple tail-entities to form overlapping relations**.

**step 2**:further decompose HE and TER  extraction with a span-based tagging scheme，即model 上述2个子任务in a unified span-based extraction framework.具体地，
>1. HE extraction：在每一个head-entity的开始和结束位置标注entity type；
>2. TER extraction: 对于所有的与给定的head-entity存在关系的tail-entities，在这些tail-entities的开始和结束位置标注relation types。

+ 为增强boundary positions之间的联系，本文使用一个**hierarchical boundary tagger**，在一个级联结构中分别标注开始和结束位置，并使用一个**multi-span decoding algorithm**一起进行解码。
### 2.1 tagging scheme
> 此处为个人理解：<br>
这里主要指 span-based 的标注方案,span很好地对应了 实体&实体对间的overlapping relations situation。对于同一个head-entity，可能对应有2个tail-entity，存在2种不同的关系类型。 对于HE extraction 和 TER extraction来说，它们分别被分解为2个sequence labeling subtasks。
+ HE extraction

> 序列标注子任务1: identify the start position of one head-entity。对于一个token，如果是实体的start word，那么便标注出其对应的entity type，否则，标注为O(Outside)；

> 序列标注子任务2:identify the end position of one head-entity.
+ TER extraction for each identified head-entity,利用 span boundaries来同时抽取tail-entities和预测relationos。

> 序列标注子任务1:为 tail-entity的start word token 标注relation type。

> 序列标注子任务2:tags the end word of the tail-entity。

see the following figure：![an-example-of-proposed-tagging-scheme]()
### 2.2 hierarchical boundary tagger
在上述2个 extractors中均加入一个通用模块-HBT，即hierarchical boundary tagger。

本节中，对head-entity和tail-entity不做区分，均被视为targets。于是，从句子S中抽取带有标签l的taeget t 的概率为：
$$p(t,l|S)=p(s_{t}^{l}|S)p(e_{t}^{l}|s_{t}^{l},S)$$
其中，$s_{t}^{l}$表示 带有标签l的target t 的start index。该公式表明：*start positions的预测结果，将有益于预测end positions*。

下图显示了本文的分层标注结构,使用一个任务将每层联系起来，并使用来自low-level task的tagging results和hidden states作为high-level task的input
![an-illustration-of-model-proposed-in-this-paper]()


### 2.3 extraction system
## 3 experiments and its results
## 4 ablation study
### 4.1 analysis on joint learnig
### 4.2 analysis on overlapping relation extraction

## main papers for references
+ \[PA-LSTM]Dai et al.@2019  *joint extraction of entities and overlappiing relations using positon-attentive sequence labelling*

directly labels entities and relations according to query positions.

**downsides of this method**:
① 忽略了内部结构，比如 head entity，tail entity 和 relation之间的依存关系；
② the model has to conduct n labeling-once processes for an n-word sentence,which is time-consuming and difficult to deploy.

+ \[GraphRel]Fu et al.@2019  *GraphRel: modeling text as relational graphs for joint entity and relation extraction*

+ \[TME]Tan et al.@2019  *jointly extracting multiple triplets with multilayer traslation constrains*

+ \[MultiDecoder]Zeng et al.@2018  *extracting relational facts by an end-to-end neural model with copy mechanism*
