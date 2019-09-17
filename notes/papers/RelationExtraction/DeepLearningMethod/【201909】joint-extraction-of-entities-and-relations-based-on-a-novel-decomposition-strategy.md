[paper link](https://arxiv.org/abs/1909.04273)

**Highlights in this paper**:

1. They decompose the joint extraction task into 2 inner-related subtasks:HE extraction and TER extraction;

2. The proposed 2 subtasks aer deconstructed into several sequence labeling problems based on the span-based tagging scheme

3. following the span-based tagging scheme, They use a general module named HBT(heirarchical boundary tagger) to extract HE and TER, and propose
a multi-span decoding algorithm to adapt to the multi-target extraction task.

## abstract
prior works for joint extraction of entities and relations **suffer from**:
+ redundant entity pairs may mislead the classifiers;
+ ignoration of important inner structure in the process of extracting entities and relations
## 1 introduction
传统的pipelined methods：忽视了实体抽取和关系预测之间的relevance

feature-baesd joint models：need complicated process of feature engineering

nueral joint models：don't pay much attention to overlapping relations

## 2 methodology
### 2.1 tagging scheme
### 2.2 hierarchical boundary tagger
### 2.3 extraction system
## 3 experiments and its results
## 4 ablation study
### 4.1 analysis on joint learnig
### 4.2 analysis on overlapping relation extraction

## main papers for references
+ \[PA-LSTM]Dai et al.@2019  *joint extraction of entities and overlappiing relations using positon-attentive sequence labelling*

directly labels entities and relations according to query positions.

**downsides of this method**:
\[GraphRel]Fu et al.@2019  *GraphRel: modeling text as relational graphs for joint entity and relation extraction*

\[TME]Tan et al.@2019  *jointly extracting multiple triplets with multilayer traslation constrains*

\[MultiDecoder]Zeng et al.@2018  *extracting relational facts by an end-to-end neural model with copy mechanism*
