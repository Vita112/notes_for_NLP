title:end-to-end relation extraction using LSTMs on sequences and tree structures  Miwa et al. @2016
## abstract 
本文提出一种新的端到端神经模型来抽取实体和实体间的关系，基于RNN的模型通过在双向序列LSTM-RNNs上堆栈双向树结构LSTM-RNNs 捕获词序列和依存树结构信息。**模型可以在单个模型中使用共享参数，来联合表征实体和实体间的关系；模型在训练阶段检测实体，通过实体预训练和计划采样/轮询采样 在关系抽取任务中使用利用实体信息**。
> scheduled sampling: 训练时，网络将不再完全采用真实序列标记作为下一步的输入，而是以一个概率p来选择真实标记，以1-p来选择模型输出。p的大小在训练过程中是变化的。**the idea**：在起初阶段，网络训练不充分，p尽量选择较大的值，即尽量使用真实标签；随着模型训练越充分，p逐渐减小，尽量选择模型自己的输出。
`解决了传统神经网络中，训练时和测试时的输入不一样的问题`。
## 1 introduction
由于关系与实体信息关系密切，端到端联合实体和关系建模将有利于提升模型性能。本文通过使用基于神经网络的模型来自动学习特征。**有2种方法来表征实体和实体间的关系: RNNs 和 CNNs**，RNNs能够直接表示基本的语言学结构信息，比如词序列和成分/依存树。**对于关系分类任务，基于LSTM-RNNs的模型表现不如CNNs好，这是因为：他们大多数包含有限的语言学结构和神经结构，而且没有联合实体和关系建模**。

本文补充结合语言学结构信息，建立了基于更丰富的LSTM-RNN结构的实体和关系的端到端模型。词序列和树结构被认为时关系抽取的补充信息。**本文在词序列和依存树结构上同时建立一个端到端模型来抽取实体间的关系，模型通过使用双向序列LSTM-RNNs和双向树结构LSTM-RNNs，来在单个模型中联合实体和关系建模**：
> 1. detect entities;
> 2. 使用一个增量式编码NN结构来抽取被检测到的实体间的关系，并且，使用实体和关系标签联合更新这个NN结构的参数。**此处，模型在训练阶段加入了2个强化操作：实体预训练(entity pretraining to pretrain the entity model) 和 轮询采样(scheduled sampling to replace unreliabel predicted labels with gold labels in a certain probability在一个特定的概率内，使用金色标签替代那些不可信的预测标签).** 这些强化消除了早期训练阶段中实体检测的低性能的问题,同时使得实体信息能够进一步帮助后来的关系分类任务。
## 2 related work
关于关系分类，有以下几种方法：embedding-based models, CNN-based models, RNN-based models, incorporating shortest dependency paths into CNN based models, LSTM-RNNs models.

Kaisheng Tai et al.(2015)提出的LSTM-RNNs聚焦于自底向上的信息传递的方向，模型不能像在类型依赖树中那样处理任意数量的类型化子级。

some existing feature-based models:
> 1. integer linear programming 整数线性规划
> 2. card-pyramid parsing 卡片金字塔解析
> 3. global probabilistic graphical models 全局概率图模型
## 3 model
model in this paper is as follows:
![end-to-end_neural_relation_extraction_based_on_bidirectional_sequential_and_tree-structured_LSTM-RNNs]()
模型主要有3个表征层组成：
> 1. 词嵌入层
> 2. 基于LSTM-RNN 的词序列层
> 3. 基于LSTM-RNN 的依存子树层

在解码阶段，在序列层上建立贪心的 自左向右的实体检测，并且在依存层发现了关系分类，依存层中的每一个基于LSTM-RNN的子树 对应于 检测出的2个实体间的一个关系候补。在解码整个模型结构后，通过时序反向传播(backpropagation through time)同步更新参数，**依存层在序列层上进行堆栈，因此，实体检测和关系分类可以共享embedding and sequence layers，并且共享的参数受到实体和关系标签的影响**。
### 3.1 embedding layer-handle embedding representations
n_w, n_p, n_d, n_e-dimensional vectors $v^{(w)},v^{(p)},v^{(d)},v^{(e)}$ are embedded to words, POS tags, dependency types, entity labels, respectively.
### 3.2 sequence layer-represent words in a linear sequence using the representations from the embedding layer
**fig. 1 中的左下角的方框区**。这一层表示句子上下文信息，并维护实体。模型使用双向LSTM-RNNs表示一个句子中的词序列。
> 句子中第t个单词的LSTM unit 由$n_{ls}$维向量的集合组成，包含：一个输入门$i_{t}$, 一个遗忘门$f_{t}$, 一个输出门$o_{t}$, 一个记忆细胞$c_{t}$, 一个隐藏状态$h_{t}$.**该LSTM unit接受一个n维输入向量xt，上一个隐藏状态$h_{t-1}$，以及上一个记忆细胞$c_{t-1}$，使用如下公式计算新的向量**：
$$i_{t}=\sigma (W^{(i)}x_{t}+U^{(i)}h_{t-1}+b^{(i)}),\\\\
f_{t}=\sigma (W^{(f)}x_{t}+U^{(f)}h_{t-1}+b^{(f)}),\\\\
o_{t}=\sigma (W^{(o)}x_{t}+U^{(o)}h_{t-1}+b^{(o)}),\\\\
u_{t}=tanh(W^{(u)}x_{t}+U^{(u)}h_{t-1}+b^{(u)}),\\\\
c_{t}=i_{t}\odot u_{t}+f_{t}\odot c_{t-1},\\\\
h_{t}=o_{t}\odot tanh(c_{t})$$

其中，σ代表sigmoid function，W 和 U 是权重矩阵，b是偏置向量；LSTM unit接收word 和 POS embeddings 的拼接 作为其输入向量：
$$x_{t}=\[v_{t}^{(w)};v_{t}^{(p)}]$$
拼接每个word对应的2个方向的LSTM units的隐藏状态向量，作为它的输出向量：
$$s_{t}=\[\overrightarrow{h_{t}};\overleftarrow{h_{t}}]$$
并且，将它喂给后续层subsequent layers。
### 3.3 entity dectection-将实体检测看作一个序列标注任务
使用encoding scheme BILOU(begin, inside, last, outside, unit)为每个word分配一个entity tag，每个entity tag代表一个实体类型和这个实体中word的位置。

在序列层的顶部进行实体检测，使用一个 带有
### 3.4 dependency layer
### 3.4 stacking sequence and dependency layers
### 3.6 relation classification
### 3.7 training
## 4 results and discussion
### 4.1 data and task setttings
### 4.2 experimental setttings
### 4.3 end-to-end relation extraction results
### 4.4 relation classification analysis results
## 5 conclusion
## reference
1. Qi Li and Heng Ji@2104 **incemental joint extraction of entity mentions and relations**ACL

2. Makoto Miwa and Yutaka Sasaki@2014 **Modeling joint entity and relation extraction with table representation**ACL

3. Kaisheng Tai et al.@2015 **improved semantic representation from tree-structured LSTM networks**ACL

4. Kun xu et al.@2015a **semantic relation classification via convolutional neural networks with simple negative sampling**ACL

5. Yan Xu et al.@2105b **classifying relations via long short term memory networks along shortest dependency paths**ACL
