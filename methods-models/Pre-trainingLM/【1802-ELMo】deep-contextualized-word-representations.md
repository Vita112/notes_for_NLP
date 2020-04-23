[paper link](https://arxiv.org/pdf/1802.05365v1.pdf)

To be presented at NAACL 2018

## abstract
models ①complex characterisitics of word use(e.g. syntax and semantics); ②how these uses vary across linguistic contexts.

本文得到的word vectors是深层双向语言模型内部状态的学习函数，在一个大型文本语料库上重新训练。
这些词表示 可以简单地添加到已有的模型中，且在6个挑战性的NLP任务中，得到了显著的性能提升。
## introduction
## 2 related work
## 3 ELMo:embeddings from language models
### 3.1 bidirectional language models
### 3.2 ELMo
### 3.3 using biLMs for supersived NLP tasks
### 3.4 pre-trained bidirectional language model architecture
## 4 evaluation
* Question answering
* textual entailment
* semantic role labeling
* coreference resolution
* named entity extraction
* sentiment analtsis
## 5 analysis
### 5.1 alternate layer weighting schemes
### 5.2 where to include ELMo?
### 5.3 what information is captured by the biLM's representations?
### 5.4 sample efficiency 
### 5.5 visualization of learnd weights
## 6 conclusion

