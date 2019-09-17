BERT-pre-training_of_deep_bidirectional_Transformers_for_language_understanding

google AI language

[paper link](https://arxiv.org/pdf/1810.04805.pdf)
## abstract
提出一种新的语言表示模型-BERT:Bidirectional Encoder Repressentations from Transformers,**旨在通过基于所有层的左、右上下文，to pretrain deep bidirectional representations from unlabeled text**.是对fine-tuninig based approaches的改进。

可以仅在一个额外的输入层上进行fine-tuning，便可以在很多NLP任务上获得优秀的表现。实验证明：BERT在11项NLP任务中获得了state-of-the-art的表现。

## 1. introduction
+ **NLP task**
> sentence-level tasks：natural language inference，paraphrasing——通过整体分析所有句子，来预测句子间的relationships；

> token-level tasks:NER, QA

+ 将预训练语言模型应用于下游任务的2种策略：feature-based 和 fine-tuning.

feature-based：例如 ELMo from [paper](https://arxiv.org/pdf/1802.05365.pdf)

fine-tuning：例如 OpenAI GPT(Generative Pre-trained Transformer)

两种方法在预训练阶段使用相同类型的objective function，即 使用单向语言模型unidirechtional language models来学习通用语言表示。这种标准的语言模型的主要限制在于：单向模型限制了 预训练阶段的构架选择。

+ **contributions in this paper**

1. 提出BERT，使用MLM(masked language model)来预训练深度双向表示；提出了NSP：next sentence prediction。

2. 表明 预训练表示弱化了 对许多精心设计的特定于某个任务的构架的需求。

3. code and pre-trained models are available：https://github.com/google-research/bert.

4. **知乎文章[《从word embedding到BERT》](https://zhuanlan.zhihu.com/p/49271699)对BERT的来龙去脉进行了梳理(此处点赞，看完后有种通了的感觉)**，参见文章末尾附录。

## 2 related work
### 2.1 unsupervised feature-based approaches
**ELMo** extract context-sensitive features from a left-to-right and a right-to-left language model.每一个token的上下文表示是这两个方向的表示的拼接。

model is feature-based and not deeply bidirectional。向量拼接这种形式只能捕获浅层特征。

### 2.2 unsupervised fine-tuning approaches
近期，可以从unlabelled text中预训练 产生contextual token representations的 sentence or document encoders，然后，进行微调用于a supervised downstream task。基于微调的方法的优点是:只需要学习少量的参数。基于此，GPT got state-of-the-art results on many sentence-level tasks。

相当于原始Transformer architecture的decoder，因其使用的是 constrained self-attention，所有的token都只关注其左侧的上下文信息。
### 2.3 transfer learning from supervised data
使用大规模数据，可以从监督任务中进行高效的迁移学习。CV领域的研究以表明：从大型预训练模型中迁移学习的重要性。

## 3 BERT：pre-training and fine-funing
> *step1-pre-training*:model is trained on unlabelled data over different pre-training tasks

> *step2-fine-tuning*: model is first initialized with pre-trained parameters,then all of the parameters are fine-tuned using labelled data from the downstream tasks.

BERT的一个区别于其他模型的特征是：**BERT的跨不同任务的统一的框架，在预训练阶段和微调阶段的模型之间只有很小的差别**。

+ **model architecture** 

![overall_pre-training_and_fine-tuning_procedures_for_BERT](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/img/overall_pre-training_and_fine-tuning_procedures_for_BERT.png)
a multi-layer bidirectional Transformer encoder based on the original Transformer in 《attention is all you need》。

+ **input/output representations**

*a sentence*:连续文本的任意span，而不是一个真正意义上的语言学上的句子；*a sequence*:BERT的输入token序列，可能是一个句子，也可能是打包在一起的2个句子。

本文使用带有30,000个token的vocabulary的 Wordpiece embeddings。每一个sequence的第一个token总是一个特殊的分类token(【CLS】)，对应于这个token的最终隐藏状态 被看作聚集序列表示，用于分类任务。input embeddings是
token embeddings，segmentation embeddings和position embeddings的加和，即下图：
![BERT_input_representations](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/img/BERT_input_representations.png)

### 3.1 pre-training BERT
pre-train BERT using 2 unsupervised tasks
+ TASK #1-Masked LM

在预训练时，使用【mask】这个特殊的token 随机替代句子中的部分token，目标是 预测【mask】对应位置的单词。这个过程又被看做cloze任务，该任务中对应masked token的最终隐藏向量are fed into一个在词汇表上的output softmax 分布。

由于在fine-tuning 阶段并没有出现【mask】token，这导致pre-training阶段和fine-tuning阶段产生了 mismatch。为缓解这种不匹配，在与训练阶段，我们并不总是使用实际的【mask】token来代替被遮蔽的words。具体地，模型首先随机选取15%的token positions用于预测，然后对于被选中的第i个token，模型有80%的概率会使用【mask】token 来替换，10%的概率使用一个随机token替换，10%的概率选择保持不变。
+ TASK #2-Next Sentence Prediction

为训练一个可以理解句子间关系的模型，我们预训练一个 二值化的下一个句子预测任务（binarized next sentence prediction task），该任务可以从任意的单语语料库中生成。当为每一个预训练样本选择句子A和句子B时，50%的时候句子A的后一句真的是句子B，50%的时候不是这样。5.1的消融分析部分证明了 **预训练这个NSP任务对于QA & NLI 任务十分有益**。

pre-training data 使用BooksCorpus的800M words和English Wikipedia。
### 3.2 fine-tuning BERT
对于每一个任务，我们将任务特定的input和output放入BERT中，然后端到端地微调所有参数。
## 4 experiments
### 4.1 GLUE:general language understanding evaluation benchmark，sequence-level task

### 4.2 SQuAD v1.1:Stanford Question Answering Dataset，token-level task
是一个 100k众包 Q/A pairs的集合。给定一个问题和 来自维基百科的包含答案的 一个段落，任务是：预测在passage中的answer text span。

将input question 和 passage表示为 a single packed sequence；在微调阶段，仅引入开始向量S 和 结束向量E；将Ti 和 S 点乘后，使用softmax计算
wi成为 answer span首个单词的概率：
$$P_{i}=\frac{e^{ST_{i}} }{\sum_{j}e^{ST_{j}}}$$
answer span的结束单词的概率计算同上类似。
### 4.2 SQuAD v2.0
### 4.3 SWAG: Situation With Adversarial Generation dataset
数据集包含113k sentence-pair completion examples，用于评估基于常识的推理。给定视频字幕数据集中的一个句子，任务是从4个选项中决定最合理的候选句子。

微调阶段，我们构建4个input sequence，每个input sequence都包含给定句子（sentence A）和可能的后续句子（sentence B）的拼接。引入新参数向量 V∈R^H，它和最终聚集表示C_i∈R^H 的点积的softmax分布，就是每一个选择的得分。
## 5 ablation study
### 5.1 effect of Pre-training tasks
 
### 5.2 effect of Model size
## 6 conclusion



## 附录
bert如此火的原因：在NLP各项任务中的出色表现 和 模型的广泛通用性。

1. **历史沿革**

+ 1.1 **从图像领域的预训练 到自然语言的预训练**

> 图像领域的预训练

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
>> **传统语言模型(非学术解释)**

语言模型假设一种语言中所有可能的句子服从一个概率分布，每个句子出现的概率加起来为1，它使用一个函数P，该函数根据句子中某个词之前的一系列单词，来预测该词位置上出现某个特定单词的概率，**句子中，每个单词都有一个根据上文信息预测自己的过程，把所有预测这些单词的概率乘起来，数值越大，代表这句话越接近自然语言，即越像一句人话**。好的语言模型应该得到较高的概率，perplexity更低，越合乎自然语言表达。**语言模型仅对句子出现的概率进行建模，并不尝试去理解句子的内容含义**。根据链式法则，该函数表示为：
$$p(S)=p(w_{1},w_{2},\cdots ,w_{m})=p(w_{1})p(w_{2}|w_{1})p(w_{3}|w_{1}w_{2})\cdots ,p(w_{m}|p(w_{1},\cdots ,w_{m-1}))=\prod_{i=1}^{m}p(w_{i}|w_{1},\cdots ,w_{i-1})$$
语言模型能够帮助从大量自然语言事实中提取出语言学知识，提高NLP中各下游任务的表现。本世纪初最流行的统计语言模型是N-gram模型，是典型的基于稀疏表示(sparse representation)的语言模型;这类**sparse representation based语言模型遭遇 数据严重稀疏的问题**。因为句子长度不固定，若是每次预测概率都考虑该词以前的所有词，模型将用到更多的参数，且为获得更准确的概率分布，需要庞大的语料支持，这将**导致参数空间过大**；加之，zipf定律：语言中大多数词在语料中的出现是稀疏的。训练样本的不足，也会大大削弱模型的可靠性。
>>> 基于马尔科夫假设的N-gram模型：一个词的出现仅依赖于它前面的n个词。

模型公式简化为：
$$p(S)=p(w_{1},w_{2},\cdots ,w_{m})=\prod_{i=1}^{m}p(w_{i}|w_{i-n+1},\cdots ,w_{i-1})$$
其中，n代表当前单词依赖它前面单词的个数。当n取1,2,3时，分别为 `unigram`,`bigram`,`trigram`。研究人员使用n=3时，获得了较好的效果，但也存在2个缺陷：`1.没有考虑超过2个单词以上的上下文信息,无法处理更长程的context(N>3);；2. 没有考虑单词之间的similarity`. n-gram模型参数一般采用极大似然估计方法计算：
$$p(w_{i}|w_{i-n+1},\cdots ,w_{i-1})=\frac{Count(w_{i-n+1},\cdots ,w_{i-1},w_{i})}{Count(w_{i-n+1},\cdots ,w_{i-1}}$$
其中，count(x)表示短语x在语料中出现的次数。
关于N-gram模型，请参考笔记[N-gram_model](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/n-gram%20model.md)

随着deep learning的崛起，以词向量(word embedding)为代表的分布式表示(distributed representation)的语言模型取得更好的效果，深刻影响着NLP领域的模型及其应用。
>> **神经网络语言模型**

NNLM（neural network language model）是基于神经网络的语言模型，其利用神经网络的非线性拟合能力，同时学习每个单词的分布式表示word embedding，以及单词序列的概率函数。Bengio于2003年**提出Probabilistic feedforward neural network language model**，改进了n-gram模型，使用依赖于共享参数的多层神经网络，可以利用更长的上下文。论文**提出分布式表示：存在一个分布式假设，即 如果2个词的上下文相同，那么，这两个词的representation也是相似的。这使得模型得以泛化。分布式表示可理解为 一种用来得到word representation的方法**。例如，知道猫和狗扮演类似的角色(语义和语法)，那么，可以将“the cat is walking in the bedroom”泛化到“a dog was running in a room”。
    
标准的NNLM构架图如下：
    ![architecture_of_NNLM](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/img/architecture_Of_NNLM.jpg)
    
模型包含input，projection，hidden和output四层，可概括为2部分：

> 1. 特征映射：通过映射矩阵C∈R(形状为 |V|×m)，将输入的每个单词映射为一个实值向量，每一行表示某个单词的特征向量，m为特征向量的维度；
> 2. 计算条件概率分布：单词的概率函数用C表示，一种方法是：使用一个函数g(g是前馈，或递归神经网络),将上下文中的单词的特征向量序列 映射到词汇表V中单词上的概率分布。

具体地，在input层中，原始单词输入wi是一个one-hot编码向量，维度为词库大小|V|；然后，乘以映射矩阵后得到表示向量c(wi)，维度为300，每个单词的c(wi)拼接，得到神经网络的输入层，假设窗口滑动大小为4；上接隐层(隐层神经元数量为100个，加激活函数tanh等，进行非线性映射)；最后，在output层，使用softmax来预测 某个特定单词是$w_{i+1}$的概率。

>>> **接下来，是跟具体的网络结构细节**：
> 1. 每次从语料库中滑动4个数据，前3个词转为one-hot编码，喂入网络；
> 2. 在映射层，根据one-hot向量中值为1的位置对应矩阵C，得到该词的表示向量，拼接w1，w2和w3的表示向量，得到词段的表示向量矩阵(300\*3)；
> 3. 在隐层，进行全连接加权求和后(3\*100)，使用tanh非线性激活函数，得到隐层结果；
> 4. 在输出层，同样进行全连接(100\*8w)，使用softmax函数，得到最后的概率输出。公式为：$$p((w_{i}|w_{i-n+1},\cdots ,w_{i-1}))=\frac{e^{y_{w_{i}}}}{\sum \_{t}e^{y_{t}}}$$
其中，$y=b+Wx+Utanh(d+Hx)$, 模型参数Θ=(b, d, W, U, H, C)。H是输入层到隐层的权重矩阵，大小为3\*100,U是隐层到输出层的权重矩阵，大小为8w\*100.

**模型的权重参数 和 词向量是同时训练的，训练完成后，同时得到 网络的权重参数和词向量**。
>>>**训练目标是优化以下损失函数，使用梯度下降算法，更新参数**：

$$L(\theta )=\frac{1}{T}\sum \_{i}logf(w_{i},w_{i-1},\cdots ,w_{i-n+1};\theta )+R(\theta )$$

### 思考：网络中，隐层的作用是什么？
>> **word embedding**

从上一节NNLM的训练过程，我们发现 网络不仅能够根据上文预测后接的单词是什么，同时，收获了一个副产品，即word embedding。2013年，最火的用语言模型做word embedding的工具是word2vec，它有2种训练方法：①CBOW，核心思想是：从一个句子中扣掉一个词，使用该词的上文(context before)和下文(context after)来预测被抠掉的词；②Skip-gram：输入某个单词，要求网络预测该单词的上下文。**word2vec与NNLM不同，NNLM的任务是 学习一个解决LM任务的网络结构，要求根据上文，预测下文；而word2vec不一样，它的目的就是生成word embedding。一个单词表达为word embedding后，很容易找出语义相近的其他词汇**。

关于word2vec工具，请参见笔记[word_embedding](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/word_embedding.md),在此不做详述。









