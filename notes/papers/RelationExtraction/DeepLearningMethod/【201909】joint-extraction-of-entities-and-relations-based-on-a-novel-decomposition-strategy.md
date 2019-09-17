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

see the following figure：

![an-example-of-proposed-tagging-scheme](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/an-example-of-proposed-tagging-scheme.jpg)
### 2.2 hierarchical boundary tagger
在上述2个 extractors中均加入一个通用模块-HBT，即hierarchical boundary tagger。

本节中，对head-entity和tail-entity不做区分，均被视为targets。于是，从句子S中抽取带有标签l的taeget t 的概率为：
$$p(t,l|S)=p(s_{t}^{l}|S)p(e_{t}^{l}|s_{t}^{l},S)$$
其中，$s_{t}^{l}$表示 带有标签l的target t 的start index。该公式表明：*start positions的预测结果，将有益于预测end positions*。

下图的右侧部分（HBT）显示了本文的分层标注结构,使用一个任务将每层联系起来，并使用来自low-level task的tagging results和hidden states作为high-level task的input.

![an-illustration-of-model-proposed-in-this-paper](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/an-illustration-of-model-proposed-in-this-paper.jpg)
+ 使用BiLSTM作为base encoder

+ HBT_HE:当标注start position时,预测单词xi的标签sta_tag(xi),如下：

![HBT_HE](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/HBT_HE.jpg)
+ HBT_TER:预测单词xi的end tag,如下:

![HBT_TER](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/HBT_TER.jpg)
此处，在BiLSTM_end层中，引入了位置嵌入信息$p_{i}^{se}$,它通过在一个可训练的position embedding matrix中查找得到，即

![p_i^se](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/p_i%5Ese.jpg)
+ 定义training loss of HBT:真正的strat和end tags的负log概率的加和：

$$L_{HBT}=-\frac{1}{n}\sum_{i=1}^{n}(logP(y_{i}^{sta}=\hat{y}\_{i}^{sta})+logP(y_{i}^{end}=\hat{y}\_{i}^{end}))$$
其中，$\hat{y}\_{i}^{sta}$和$\hat{y}\_{i}^{end}$是第i个单词真正的start和end tags。

+ use multi-span decoding algorithm to adapt to the multi-target extraction task

![multi-span decoding algorithm](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/multi-span%20decoding%20algorithm.jpg)
### 2.3 extraction system
使用span-based tagging scheme 和 hierarchical boundary tagger，本文提出一种end-to-end neural architecture 来联合抽取实体和重叠关系。
+ Shared Encoder

input word representation 包括pre-trained embeddings 和 在单词xi的character
sequence上使用CNN得到的character-based word representations。
+ HE Extractor

拼接hi和g得到feature vector $\tilde{x_{i}}=\[h_{i};g]$,其中，hi是an input token representation，g是在所有hidden states上进行max pooling后得到的global contextual embedding。

将H_HE喂入HBT来抽取head-entities：
$$H_{HE}={\tilde{x_{1}},\cdots ,\tilde{x_{N}}}$$
$$R_{HE}=HBT_{HE}(H_{HE}),R_{HE}={(h_{j},type_{h_{j}})}\_{j=1}^{m}$$
$R_{HE}$包含了句子S中所有的head-entities 和 对应的entity type tags。
+ TER Extractor

$$\tilde{x_{i}}=\[h_{i};g;h^{h};p_{i}^{ht}]$$
其中，$h^{h}=\[h_{sh};h_{eh}]$指 representation of head-entity h，是h在start index和end index上的hidden state 的拼接。$p_{i}^{ht}$是position embedding，编码了从当前word xi到h的relative distance。
$$H_{TER}={\tilde{x_{1}},\cdots ,\tilde{x_{N}}}$$
$$R_{TER}=HBT_{TER}(H_{TER}),R_{TER}={(t_{o},rel_{o})}\_{o=1}^{z}$$
+ training of joint extractor

joint loss is as follow:
$$L=\lambda L_{HE}+(1-\lambda )L_{TER}$$
## 3 experiments and its results
DATASETS
+ NYT-single

+ NYT-multi

+ wiki-kBP

EVALUATION
+ a triplet is marked correct when its relation type and 2 corresponding entities are all correct

RESULTS

![main-results-on-3-benchmark-datasets](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/main-results-on-3-benchmark-datasets.jpg)
## 4 ablation study
### 4.1 analysis on joint learnig
HE extractor and TER extractor work in the mutual promotion way,which again confirms the effectiveness an rotionality of the decomposition strategy used in this paper.
### 4.2 analysis on overlapping relation extraction
我们将NTY-multi的test set分为3个categories：Normal，singleEntityOverlap，EntityPairOverlap。
+ **Normal**：there is no triplets in a sentence has overlapping entities;

+ **EntityPairOverlap**:the entity pairs of 2 triplets are identical but the relations are different;

+ **singleEntityOverlap**:some of triplets in a sentence have an overlapped entity and these triplets don't have overlapped entity pair

**本文的模型并没有解决entity pair overlapping problem**。

## main papers for references
+ \[PA-LSTM]Dai et al.@2019  *joint extraction of entities and overlappiing relations using positon-attentive sequence labelling*

directly labels entities and relations according to query positions.

**downsides of this method**:
① 忽略了内部结构，比如 head entity，tail entity 和 relation之间的依存关系；
② the model has to conduct n labeling-once processes for an n-word sentence,which is time-consuming and difficult to deploy.

+ \[GraphRel]Fu et al.@2019  *GraphRel: modeling text as relational graphs for joint entity and relation extraction*

+ \[TME]Tan et al.@2019  *jointly extracting multiple triplets with multilayer traslation constrains*

+ \[MultiDecoder]Zeng et al.@2018  *extracting relational facts by an end-to-end neural model with copy mechanism*
