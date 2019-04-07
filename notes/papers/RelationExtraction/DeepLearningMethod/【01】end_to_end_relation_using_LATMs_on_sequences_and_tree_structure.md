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
## 3 model
### 3.1 embedding layer
### 3.2 sequence layer
### 3.3 entity dectection
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
