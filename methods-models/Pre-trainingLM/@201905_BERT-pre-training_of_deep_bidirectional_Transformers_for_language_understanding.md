BERT-pre-training_of_deep_bidirectional_Transformers_for_language_understanding

google AI language

## 摘要
提出一种新的语言表示模型-BERT:Bidirectional Encoder Repressentations from Transformers,**旨在通过基于所有层的左、右上下文，to pretrain deep bidirectional
representations from unlabeled text**.是对fine-tuninig based approaches的改进。

可以仅在一个额外的输入层上进行fine-tuning，便可以在很多NLP任务上获得优秀的表现。实验证明：BERT在11项NLP任务中获得了state-of-the-art的表现。

## 1. introduction
+ **NLP task**
> sentence-level tasks：natural language inference，paraphrasing——通过整体分析所有句子，来预测句子间的relationships；

> token-level tasks:NER, QA

+ 将预训练语言模型应用于下游任务的2种策略：feature-based 和 fine-tuning.

feature-based：例如 ELMo

fine-tuning：例如 OpenAI GPT(Generative Pre-training Transformer)

两种方法在预训练阶段使用相同的objective function，即 使用单项语言模型unidirechtional language models来学习通用语言表示。这种标准的语言模型的主要限制在于：
单向模型限制了 预训练阶段的构架的选择。

+ **本文贡献**

1. 提出BERT，使用MLM(masked language model)来预训练深度双向表示。

2. 表明 预训练表示减少了 对许多精心设计的特定于某个任务的构架的需求。

3. code and pre-trained models are available：https://github.com/google-research/bert.

4. **以下来自知乎文章[从word embedding到BERT](https://zhuanlan.zhihu.com/p/49271699)的理解和整理**，参见文章末尾附录。




## 附录
bert如此火的原因：在NLP各项任务中的出色表现 和 模型的广泛通用性。

1. **历史沿革**

+ 1.1 从图像领域的预训练 到自然语言的预训练

> 图像领域的预训练：底层特征的可复用性和高层特征的任务相关性

过程如下：
```
1. 使用训练集合A 或者集合B，预训练网络，学会网络参数，留存以备用；
2. 当面临任务C时，采用相同的网络结构，浅层CNN结构的参数初始化时，使用之前训练好的参数；训练网络时，有2中方法：Frozen，即浅层加载的参数在训练过程中保持不变；
另一种是fine-tuning，初始化时使用预训练参数，在C任务的训练过程中，参数不断调整。
```
**优点**：任务C的训练数据较少，而好用的网络构架的层数却很深时，使用预训练好的参数来初始化网络构架参数，然后，使用任务C的
训练数据进行fine-tuning，可加快任务训练的收敛素的，得到较好的效果。








