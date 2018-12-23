 一种基于深度学习的实体关系抽取方法及应用  周亚林  浙大硕士论文 2018年
 ## 1.abstract
 提出2种基于深度学习的关系抽取方法：
 >①BLSTM + Attention + 浅层句法特征和实体特征
 >②使用word level CNN 和 character level CNN，抽取句子特征，结合2种结果进行预测
 对deepdive关系抽取工具进行二次开发，在原有数据基础上，加入数据增量处理、基于深度学习的训练、预测模块。
 
 
 ## 2.先行研究
 从 实体关系分类、实体和关系联合抽取、开放式关系抽取 3方面介绍。
 ### 2.1 实体间关系分类
 pipeline task：先抽取句子中的实体，再对实体对的关系进行分类

+ 基于特征向量的方法——从实体对的上下文中抽取`词语、词性、实体类别、最短依赖路径`等特征，训练分类器进行分类
>研究者有：Kambhatla使用实体本身 实体类型 句法解析树等，训练最大熵模型(ME)进行关系分类；Zhou提取更多句法解析树；
>Sun提出基于词语聚类的特征抽取方法，选择聚类效果较好的词语集合进行分类
+ 基于树核函数的方法——定义核函数，在句子的浅层句法分析上计算2个句子的相似度，使用SVM进行关系分类。
>研究者有：Bunescu提出基于`最短依存句法路径`的核函数，计算2个句子在依存句法路径上的`相同词语个数`，来计算`句子相似度`；
>Zhang提出`卷积树核函数`，提取蕴含在句法树中的`句法信息`。
+ 基于深度学习的方法
```
socher提出矩阵-向量递归神经网络(MV-RNN)来捕捉句子中复杂的语义信息。
zeng将词语、词语的相对位置特征作为输入，使用CNN提取sentence level的特征，结合entity context和wordnet上位词语等特征。
santos提出基于排序的CNN模型，为每个类别定义一个标准特征向量，同CNN生成的句子特征向量做内积，内积最大的类作为最终预测结果。
xu先假设：依存关系树中2个实体间最短路径上的词语描述了这2个实体间的关系，然后使用CNN对最短路径上的文本提取特征，并分类。
yan提出在最短依存路径上使用LSTM模型，将路径上的词语、pos tagging、语法关系、wordnet上位词作为输入。
```
attention注意力机制应用于关系抽取，提出端到端的模型。
```
zhou 将句子中的词语作为输入，使用BLSTM处理后的输出，作为Attention的输入，再次处理。
wang 使用2层注意力机制的 CNN 模型，首先为每一个类别定义一个特征向量，使用2个实体 对输入句子中的词语 进行1次attention；然后用CNN对句子进行特征抽取，
对 抽取结果使用 类别特征向量进行 第2层attention。
```
distance supervision远程监督方法
>mintz 使用distance supervision 生成带噪音的标注数据集，在极短时间内生成大量标注样本，缓解数据不足的问题。**对于噪音标注问题**，<br>
zeng 提出了multi-instance的 nn模型，在训练时，对于每个实体对，只选取最能反映其关系的句子来训练。<br>
lin 在 sentence level上使用 attention，对同一对实体的不同句子 分配 不同的权重，减少噪音样本的影响。
### 2.2 实体和关系联合抽取——参数共享和标注策略
+ zheng 使用BLSTM对句子进行编码，分别使用一个LSTM进行NER，一个CNN进行关系分类，分类时根据NER的结果选取实体对。模型共享底层的BLSTM。
+ miwa 同时使用BLSTM(用于实体检测)和树状LSTM(用于关系抽取)对句子进行建模，2者参数共享。
+ zheng 提出一种新的标注策略：新标签既包含实体相关信息，同时对关系进行了编码，将问题转化为一个`序列标注任务`。
### 2.3 开放域实体关系抽取
+ yates textrunner系统，利用启发式规则自动建立关系抽取语料，使用简单的句法特征训练朴素贝叶斯分类器，进行关系抽取，利用大规模预料的冗余性提高结果的precision。
+ wu 实现了woe系统，利用维基百科的infobox数据来生成标注语料，提高关系抽取的precision和recall。
+ etzioni 提出将动词作为指示词的 reverb系统，减少错误抽取的比例。
+ mausam 进一步开发了ollie系统，以reverb抽取的高质量三元组作为种子，通过bootstrapping 迭代生成大量training data，学习出一批开放的规则模板
基于聚类的无监督方法：假设上下文相似的词语具有相近的含义，那么若实体对的上下文信息类似，则很有可能属于同一种关系。
> hasegawa 通过对命名实体间的文本进行聚类，自动对实体关系进行识别，聚类后的每一个类簇对应一种关系类型，将类别中频率最高的词语作为关系描述词。
> zhang 利用浅层句法树计算相似度，通过`层次聚类`进行re，兼顾低频实体对间的语义关系
**缺点**：难以对每个类簇生成合适的标签；很多结果没有实际含义
## 3 关键技术概述
### 3.1 卷积神经网络
CNN专门用来处理网络状拓扑结构的输入，例如图片或文本序列；是一种稀疏连接的网络，每个输出节点只与局部的输入节点相连。
+ 卷积核

是卷积层的参数矩阵。进行计算时，卷积核与输入的各区域进行`点积`计算，将点积结果作为该位置的输出。图示如下：

![文本的卷积](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/WenBenJuanJi.png)

**局部连结性使得运算量减少；各位置使用同一个卷积核，减少了参数个数，不易造成过拟合。**
+ 池化操作

最常用的是 最大池化max pooling，将图片划分成互不重叠的一个个矩形区域，取 每个区域中的最大值 作为输出，最后输出一个矩阵.
**是对输入的一种归纳；使得网络对局部的微小变化具有不变性**
### 3.2 循环神经网络
RNN是一种用于处理序列数据的神经网络，它将网络的上一个状态作为当前计算的输入，使得整个网络具有一定记忆功能，利用之前信息。结构如下：

![RNN](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/RNN.png)
虽然理论上，RNN可以利用之前所有的历史信息，但链式法则求导后，在激活函数的作用下， 多个绝对值小于1的数相乘，梯度呈指数级的趋向于0.<br>
**梯度消失**：在长程依赖关系中，后面的梯度很难传递到前面的位置。
+ LSTM——利用自适应的门gate 来控制信息传递

LSTM单元结构图如下：

![LSTM单元](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/lstm_unit.png)
LSTM单元包括三个门：input gate，forget gate，output gate，计算公式如下:

![calculation_formula](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/calculation_formula.png)
此外，有一个记忆单元memoryCell，用于保存信息，是lstm网络中，信息长距离传输的通道。记忆单元的计算公式如下：

![calculation_formula_for_memory_cell](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/calculation_formula_for_memory_cell.png)
+ 双向LSTM——使用2个独立的LSTM网络分别`从前往后`和`从后往前`处理整个序列。

BLSTM在位置i的隐藏状态为：$$h_{i}=\[\overrightarrow{h_{i}},\overleftarrow{h_{i}}]$$
### 3.3 DeepDive
斯坦福大学NLP组开发的`知识抽取工具`，用于`从文本中抽取结构化数据`。采用`因子图`作为关系抽取模型，用户对特征抽取部分进行定义后，deepdive会自动将各个模块进行整合，形成一套`端到端的关系抽取系统`。
开发过程略
## 4 数据获取与标注
对文本进行ner,抽取文本中的公司实体；对实体两两配对，生成`待分类实体对`，对样本进行打标；划分training dataset和test dataset。
+ NER

工具：LTP，支持分词、词性标注、NER、句法分析功能
实现：先 分词和词性标注，然后 将结果传递给 实体命名识别工具。使用BIES标签标识方法检测实体。对于错误标注和遗漏，需要手动编写规则进行补充，比如`正则表达式`。
## 5 基于RNN 的关系抽取模型
![RNNBasedMethod](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/RNNBasedMethod.png)
+ feature extraction
主要特征有：
```
1. 句子中的词语：使用jieba进行分词提取
2. 词性：使用LTP的pos tagging进行词性标注
3. NER:使用LTP的 NER工具
4. 相对于2个实体的位置：将每个词相对于2个实体的位置作为特征
5. 实体左右两边的2个词语：单独抽取实体的上下文，将这些词的embedding向量拼接起来，输入给最后一层的全连接层分类器。
6. 实体对本身
```
+ embedding 层
>word embedding

核心思想：将one-hot表示转换为`稠密的向量表示`，使词向量间的空间关系很好地反映`词与词的语义关系`。采用`预训练词向量`方法来对词向量初始化。下图是是使用**word2vec训练算法**：
![word2vec](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/Masters-_thesis-1/pictures/word2vec.png)
> 其他特征的embedding

对每个pos tagging label，NER label和position label，进行embedding。结合word embedding，得到每个词的最终表示：
$$e_{i}^{all}=\[e_{i}^{word},e_{i}^{position1},e_{i}^{position2},e_{i}^{pos},e_{i}^{ner}]$$

+ BLSTM层——引入自适应的门，来控制LSTM单元是否保留前一时刻的状态，或者是否记住当前时刻的状态.结构如下：

![BLSTM]()
+ attention层
![attention]()
+ 特征融合和分类
将$h_sent$,$h_ctx$,$h_entity$拼接起来，得到结果如下图：

![feature_integration]()
## 6 基于CNN 的关系抽取模型
该模型通过卷积神经网络提取句子特征，再将句子特征与实体本身以及实体的上下文特征进行拼接，最后通过1个全连接的神经网络来对特征进行分类.
框架如下图：

![CNN]()
### 6.1 word-level CNN

先将文本进行分词，然后将词语作为句子的基本单元，在进行卷积时，每个词语就相当于图像中的像素点，同时我们将词语的词性标签以及命名实体识别标签也作为特征 ，与词向量拼接起来，一起进行卷积.
>对每个词`提取上下文信息`，使用`一个滑动窗口`，对于每个单词，`将它前后大小为k的窗口中的词拼接起来，作为该词的表示`，在句子前后进行padding，加入一定数量的占位符。

![convolutional_kernel]()

>池化层-max pooling

对于卷积层的输出矩阵C，在每个维度上求各个位置的最大值，生成一个固定长度的向量，来表示整个句子的信息。

### 6.2character-level CNN

句子是字符组成的序列，将每个字符替换为其对应的字向量，输入到CNN模型中，max-pooling综合每个位置的输出。
>卷积层

使用多个窗口大小的卷积模型，在不同的窗口宽度上使用不同的卷积核，每个卷积核分别在其对应的窗口宽度上进行卷积，最后，将多个
卷积核的结果拼接起来，作为最终的输出。
>池化层-max pooling

对每个卷积核的输出单独进行池化，再将赤化的结果拼接。
## 7 实验设计与结果分析
### 7.1 实验设计
+ 实验环境：服务器配置；python；模型使用tensorflow实现。
+ 评价指标：precision，recall，F1-值
+ 参数设置

各种embedding的维度设置
### 7.2 不同模型对比实验
### 7.3 循环神经网络模型分析
+ BLSTM效果比单向LSTM好很多
+ attention效果分析

BLSTM为句子的每一个位置输出一个特征向量，该向量是该位置上下文的高阶表示。由于句子长度通常不一样，不同位置的词语无法对齐，因此，需要对BLSTM的输出进行处理，以输出一个固定长度的向量，综合表达整个句子的信息。**有三类做法**：
```
按照效果好坏从高往低分别为：
attention：增加 那些对分类作用更强的词语的权重，削弱没有明显区分性的词语的权重，根据attention weights对BLSTM各个时间点上的
向量做加权平均。
first and last step首尾向量拼接
average-pooling
max-pooling
```
## 7.4 循环神经网络模型分析
+ CNN
结合使用word-level CNN 和 char-level CNN，可以达到最好的效果，`因为结合使用时，特征更多样化`。此外，word-level效果要比char-level效果好。char-level CNN的效果要优于 只使用文本的词语作为输入
+ Pooling:max-pooling结果比average-pooling好很多

## 7.5 CNN 与 BLSTM模型对比
+ 训练速度

+ 分类结果










 
 
 
