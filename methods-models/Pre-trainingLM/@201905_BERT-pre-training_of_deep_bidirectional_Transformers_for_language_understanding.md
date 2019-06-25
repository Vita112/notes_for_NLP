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

4. **知乎文章[《从word embedding到BERT》](https://zhuanlan.zhihu.com/p/49271699)对BERT的来龙去脉进行了梳理(此处点赞，看完后有种通了的感觉)**，参见文章末尾附录。




## 附录
bert如此火的原因：在NLP各项任务中的出色表现 和 模型的广泛通用性。

1. **历史沿革**

+ 1.1 **从图像领域的预训练 到自然语言的预训练**

> 图像领域的预训练：底层特征的可复用性和高层特征的任务相关性

过程如下：
```
1. 使用训练集合A 或者集合B，预训练网络，学会网络参数，留存以备用；
2. 当面临任务C时，采用相同的网络结构，浅层CNN结构的参数初始化时，使用之前训练好的参数；训练网络时，有2中方法：Frozen，即浅层加载的参数在训练过程中保持不变；
另一种是fine-tuning，初始化时使用预训练参数，在C任务的训练过程中，参数不断调整。
```
**优点**：任务C的训练数据较少，而好用的网络构架的层数却很深时，使用预训练好的参数来初始化网络构架参数，然后，使用任务C的训练数据进行fine-tuning，可加快任务训练的收敛素的，得到较好的效果。
> 预训练的合理性：底层特征的可复用性和高层特征的任务相关性

我们的网络是一个 由自底向上特征形成层级结构CNN的构架，不同层级的神经元学习不同类型的图像特征。假设有一个人脸识别的任务，完成网络训练后，若将每层神经元学习到的特征进行可视化，会发现:*最底层的神经元学到的是线段等特征，第二个隐层学到的是人脸五官的轮廓，第三层学到的是人脸的轮廓，这便是特征的层级结构*，**可观察到：越是底层的特征，越是所有不论什么领域的图像都会具备的底层基础特征，比如边、角、弧等，而越往上，抽取出的特征，越是与特定任务相关**。因此，预训练好的网络参数，尤其是底层的网络参数抽取出的特征，越具备通用性。在给定任务的训练数据规模很小时，使用这些训练好的底层参数进行新任务的参数初始化，将极大地提高该任务的训练效果。*对于高层特征，由于跟特定任务的关联不大，实际可不使用，或者采用fine-tuning用新任务的数据**清洗掉**这些高层抽取器抽出的特征*。

>> 用于预训练的数据集一般具备的特点：1. 规模大，数据量足够多；2. 跨领域，数据涉及范围广。

+ **1.2 NLP领域的预训练**
> word embedding-NLP早期预训练技术
>> 













