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
![end-to-end_neural_relation_extraction_based_on_bidirectional_sequential_and_tree-structured_LSTM-RNNs](https://github.com/Vita112/notes_for_NLP/blob/master/notes/papers/RelationExtraction/DeepLearningMethod/imgs/end-to-end_neural_relation_extraction_based_on_bidirectional_sequential_and_tree-structured_LSTM-RNNs.png)
模型主要有3个表征层组成：
> 1. 词嵌入层
> 2. 基于[LSTM-RNN](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/LSTM.md) 的词序列层
> 3. 基于LSTM-RNN 的依存子树层

在解码阶段，在序列层上建立贪心的 自左向右的实体检测，并且在依存层发现了关系分类，依存层中的每一个基于LSTM-RNN的子树 对应于 检测出的2个实体间的一个关系候补。在解码整个模型结构后，通过时序反向传播(backpropagation through time)同步更新参数，**依存层在序列层上进行堆栈，因此，实体检测和关系分类可以共享embedding and sequence layers，并且共享的参数受到实体和关系标签的影响**。
### 3.1 embedding layer-handle embedding representations
n_w, n_p, n_d, n_e-dimensional vectors $v^{(w)},v^{(p)},v^{(d)},v^{(e)}$ are embedded to words, POS tags, dependency types, entity labels, respectively.
### 3.2 sequence layer-represent words in a linear sequence using the representations from the embedding layer
**fig. 1 中的左下角的方框区**。这一层表示句子上下文信息，并维护实体。模型使用双向LSTM-RNNs表示一个句子中的词序列。
> 句子中第t个单词的LSTM unit 由$n_{ls}$维向量的集合组成，包含：一个输入门$i_{t}$, 一个遗忘门$f_{t}$, 一个输出门$o_{t}$, 一个记忆细胞$c_{t}$, 一个隐藏状态$h_{t}$.

> **该LSTM unit接受一个n维输入向量xt，上一个隐藏状态$h_{t-1}$，以及上一个记忆细胞$c_{t-1}$，使用如下公式计算新的向量**：

$$f_{t}=\sigma (W^{(f)}x_{t}+U^{(f)}h_{t-1}+b^{(f)}),\\\\
i_{t}=\sigma (W^{(i)}x_{t}+U^{(i)}h_{t-1}+b^{(i)}),\\\\
u_{t}=tanh(W^{(u)}x_{t}+U^{(u)}h_{t-1}+b^{(u)}),\\\\
c_{t}=i_{t}\odot u_{t}+f_{t}\odot c_{t-1},\\\\
o_{t}=\sigma (W^{(o)}x_{t}+U^{(o)}h_{t-1}+b^{(o)}),\\\\
h_{t}=o_{t}\odot tanh(c_{t})$$

其中，σ代表sigmoid function，W 和 U 是权重矩阵，b是偏置向量；LSTM unit接收word 和 POS embeddings 的拼接 作为其输入向量：
$$x_{t}=\[v_{t}^{(w)};v_{t}^{(p)}]$$
LSTM 的核心是 Cell state, 每个unit中包含4个交互的层：
> 1. forget gate:决定不把何种信息放入cell state中，sigmoid function输出一个在 0 到 1 之间的数值给每个在细胞状态$C_{t-1}$中的数字。1 表示“完全保留”，0 表示“完全舍弃”；
> 2. input gate:确定何种信息会被存放进cell state中，包含2部分：一个sigmoid层，决定需要更新的值；一个tanh层，创建一个新的候选值向量，对应上述公式中的$u_{t}$;
> 3. update Ct：将旧状态与ft相乘，丢弃确定不需要的信息；输入中决定需要更新的值乘以新的候选；将两者相加得到更新后的cell state；
> 4. output gate:使用sigmoid层确定细胞状态中哪些部分需要输出；使用tanh层对更新后的cell state进行处理；将两者的值相乘，输入最终确定的输出结果。

拼接每个word对应的2个方向的LSTM units的隐藏状态向量，作为它的输出向量：
$$s_{t}=\[\overrightarrow{h_{t}};\overleftarrow{h_{t}}]$$
并且，将它喂给后续层subsequent layers。
### 3.3 entity dectection-将实体检测看作一个序列标注任务
使用encoding scheme BILOU(begin, inside, last, outside, unit)为每个word分配一个entity tag，每个entity tag代表一个实体类型和这个实体中word的位置。

在序列层的顶部进行实体检测，使用一个 带有$n_{h_{e}}$维的隐藏层$h^{(e)}$和softmax输出层的 双层(2-layered)神经网络，用于实体检测：
$$h^{(e)}\_{t}=tanh(W^{e_{h}}\[s_{t};v_{t-1}^{(e)}]+b^{(e_{h})})\\\\
y_{t}=softmax(W^{e_{y}}h_{t}^{(e)}+b^{(e_{y})})$$

我们使用从左向右的 贪心的方法 为words分配实体标签，在这个解码过程中，我们使用当前word的预测标签来预测下一个word的标签，以便考虑到标签依存。上述双层NN 的输入是 其在sequence layer对应的输出和前一个word的label embedding的拼接。
### 3.4 dependency layer-表示依存树中一对target words之间的一个关系，负责特定关系表示
依存层**主要聚焦在 依存树中一对target words间的最短路径(在最小公共节点和2个target words之间的路径)**。

我们使用双向树结构LSTM-RNNs，通过捕获围绕在目标单词对周围的依存结构来表示一个关系候选。这个双向结构向每个node不仅传递叶子节点信息，还传递根节点信息，充分利用了结构树底部附近的自变量节点。**本文的top-down LSTM-RNN 把信息从结构树的top发送至这些近叶 子节点，模型在相同类型的children node间共享权重矩阵Us，并且允许不同数量的children**。在有C(t)children的第t个node上，我们计算LSTM unit中的$n_{l_{t}}$维向量：
$$i_{t}=\sigma (W^{(i)}x_{t}+\sum_{l\in C(t)}U^{(i)}\_{m(l)}h_{tl}+b^{(i)}),\\\\
f_{tk}=\sigma (W^{(f)}x_{t}+\sum_{l\in C(t)}U^{(f)}\_{m(k)m(l)}h_{tl}+b^{(f)}),\\\\
o_{t}=\sigma (W^{(o)}x_{t}+\sum_{l\in C(t)}U^{(o)}\_{m(l)}h_{tl}+b^{(o)}),\\\\
u_{t}=tanh(W^{(u)}x_{t}+\sum_{l\in C(t)}U^{(u)}\_{m(l)}h_{tl}+b^{(u)}),\\\\
c_{t}=i_{t}\odot u_{t}+\sum_{l\in C(t)}f_{tl}\odot c_{tl},\\\\
h_{t}=o_{t}\odot tanh(c_{t})$$
其中，m(·)是一个类型映射函数。

我们对3个结构选项进行了实验：
> 1. SP-Tree(shortest path structure): 捕获一个目标单词对之间的核心依存路径；
> 2. SubTree: 在目标词对的最低共同祖先下的子树，为SPTree中的路径和单词对提供额外的修正信息；
> 3. FullTree：完全树，捕获整个句子的上下文信息
### 3.4 stacking sequence and dependency layers
在sequence layer的顶部堆栈relation candidates对应的dependency layers，以便把word sequence 和 dependency tree structure information合并进输出中。

第t个word的dependency layer LSTM unit接收输入
$$x_{t}=\[s_{t};v_{t}^{(d)};v_{t}^{(e)}]$$
$s_{t}$指的是 在sequence layer中其对应隐藏状态向量的拼接；$v_{t}^{(d)}$指的是dependency type embedding依存类型嵌入,表示对父级依赖的类型；$v_{t}^{(e)}$指的是label embedding标签嵌入，对应于预测的实体标签。
### 3.6 relation classification
使用检测到的实体的最后单词的所有可能的组合来增量式地构建relation candidates。在fig. 1 中，通过使用 带有L-PER标签的Yates和带有U-LOC标签的Chicago，构建了一个关系候选。对每一个关系候选，我们发现依存层dp，它对应于关系候选中单词对p之间的路径；NN接收 由依存树层的输出构建的一个关系候选的向量，并预测它的关系标签。

使用类型和方向表示关系标签,不包括negative relations，因为这些关系没有方向。

**关系候选向量是一个拼接向量**（每个向量前面都带有方向箭头，第一个是↑自底向上；后面两个是↓自顶向下）
$$d_{p}=\[h_{p_{A}};h_{p_{1}};h_{p_{2}}]$$
其中，$h_{p_{A}}$是 自底向上LSTM-RNN中的最顶层LSTM单元的隐藏状态向量，表示目标单词对p的最低共同祖先；$h_{p_{1}}$ 和 $h_{p_{2}}$是2个LSTM单元的
隐藏状态向量，表示 自顶向下LSTM-RNN中的第一个和第二个目标单词。**然后，使用一个 带有$n_{h_{r}}$维的隐藏层$h^{(r)}$和softmax输出层的 双层(2-layered)神经网络,进行关系分类**：
$$h^{(r)}\_{p}=tanh(W^{(r_{h})}d_{p}+b^{(r_{h})})\\\\
y_{p}=softmax(W^{r_{y}}h_{t}^{(r)}+b^{(r_{y})})$$
**由于模型从堆栈在sequential LSTM-RNNs上的树结构LSTM-RNNs中构建input $d_{p}$, 因此，sequence layer对input $d_{p}$ 没有直接的贡献。这导致不能充分利用实体信息。为此，对于每个实体，从sequence layer到关系分类的input $d_{p}$，我们拼接隐藏状态向量的平均值**：
$$d_{p}^{'}=\[d_{p};\frac{1}{\left | I_{p1} \right |}\sum_{i\in I_{p1}}s_{i};\frac{1}{\left | I_{p2} \right |}\sum_{i\in I_{p2}}s_{i}]$$
$I_{p1} $是实体对的word indices。

预测时，我们为每一单词对分配2个标签，因为我们考虑了从左向右和从右向左两个方向，当预测标签不一致时，选择正性的、信赖性更好的标签。
### 3.7 training
使用gradient clipping，parameter averaging，L2 regularization，通过BPTT和Adam，来更新模型参数，包括weigts，biases以及embeddings。在embedding layer，实体检测和关系分类的final hidden layers中，使用dropout。

应用了2个强化手段——scheduled sampling 和 entity pretraining，已解决下面2个问题：
> 1. 消除训练早期阶段的 实体的不可靠预测问题；
> 2. 从检测实体中建立positive relation instances。

在scheduled sampling阶段，在概率$\epsilon \_{i}$之内，本文使用gold labels作为预测结果，$\epsilon \_{i}$依赖于训练次数。使用inverse sigmoid decay计算$\epsilon \_{i}$：
$$\epsilon \_{i}=\frac{k}{k+sxp(\frac{i}{k})}$$
k是一个超参，调整模型以何种频率使用gold labels作为预测结果。

在entity pretraining 阶段，模型在训练整个模型参数之前，使用训练数据预训练实体检测模型。
## 4 results and discussion
### 4.1 data and task setttingss
using ACE05 and ACE04 for end-to-end relation extraction;  using SemEval2010 Task 8 for relation classification

ACE05： 定义了7个粗粒度的实体类型, 6个粗粒度的实体间的关系类型。一个实体是正确的，当它的类型和它头部的区域是正确的；一个关系是正确的，当
他的类型和参数实体是正确的。*因此，我们将所有在错误实体上的non-negative relations视为false positives*。

ACE04： 定义了7个粗粒度的实体类型, 7个粗粒度的实体间的关系类型。

SemEval2010 Task 8:定义了9个名词性之间的关系类型，以及一个other类型（当两个名词之间没有关系）。treat OTHER type as a negative relation
type,and no direction is considered.
### 4.2 experimental setttings
使用cnn library 实现模型；使用带original stanford dependencies 的 stanford nueral dependencu parser解析文本；设置embedding dimensions n_w 为200，n_p,n_d,n_e为25， 中间层维度dimensions of intermediate layers为100，使用在Wikipedia上训练好的wrod2vec
来初始化word vectors，并随机初始化所有其他的parameters

使用开发集微调超参tune the hyper-parameters
### 4.3 end-to-end relation extraction results
为了分析我们的端到端关系抽取模型的各个部件的贡献，我们在ACE05上进行了消融测试ablation tests：
> 发现1：当我们去掉entity pretraining ，或者同时去掉scheduled sampling 和 entity pretraining时，性能显著下降(p< 0.05)

原因1：只有当两个实体都被发现时，模型才能产生关系实例，并且没有这些强化，一切将变得很迟以至于不能找到一些关系。

> 发现2：在检测实体和关系时，不考虑sharing parameters，比如embedding and sequence layers，具体的，先训练一个实体检测model；
然后使用检测到的实体创建一个独立的RE model。此时，在实体检测和关系分类中，性能都出现了轻微的下降。

> 发现3：当去掉所有的enhancements(scheduled sampling, lable embedding, shared parameters, entity pretraning),性能甚至比SP-Tree还差。


> 发现4：对于端到端关系提取任务，选择适当的输入树结构表示（这里指最短路径）比在该输入上选择何种LSTM-RNN结构更重要（即选择sequential，还是tree-based)。
>> 1. 比较不同结构的LSTM-RNN的性能。*首先，比较了LSTM-RNNs的三种不同的输入依存结构（SPTree，SubTree，FullTree）* → 当我们将最短路径上的nodes同其他的nodes区分开时，LSTM-RNNs的表现几乎是一样的。**可能是由于模型的不同仅出现在拥有多个孩子的节点上**。
>> 2. 用最短路径给出了两个对应的基于序列的LSTM RNN的结果。*SPSeq是最短路径上的双向LSTM-RNN* :  LSTM 单元接收序列层作为input，序列层与周围依赖关系类型和方向的嵌入 拼接；拼接关系候选的2个RNNs输出。*SPXu是对Xu等人提出的shortest path LSTM-RNNs的适应，以匹配我们的sequence-layer based model* ：  对于最短路径的左右两个子路径，有2个LSTM-RNNs。**这些sequence-based LSTM-RNN的比较显示：在表示最短路径时，一个树结构的LSTM-RNN与 基于序列的 LSTM-RNN 是差不多的**。
### 4.4 relation classification analysis results
使用SemEval-2010 Task 8 来分析不同LSTM结构，architecture components，以及classification task settings。

对于关系分类来说，上述 **发现4的内容 同样适用**。
## 5 conclusion
本文提出的模型 **允许只使用同一个模型来表示实体和关系**。得出以下3个关键发现：
1. 同时使用word sequence 和 dependency tree strucure十分有效；

2. 使用共享参数训练 可提高关系抽取准确性，特别是应用在entity pretraining，scheduled sampling，label embeddings

3. 最短依存路径同样适用于神经LSTM-RNN模型，来表示树结构。
## reference
1. Qi Li and Heng Ji@2104 **incemental joint extraction of entity mentions and relations**ACL

2. Makoto Miwa and Yutaka Sasaki@2014 **Modeling joint entity and relation extraction with table representation**ACL

3. Kaisheng Tai et al.@2015 **improved semantic representation from tree-structured LSTM networks**ACL

4. Kun xu et al.@2015a **semantic relation classification via convolutional neural networks with simple negative sampling**ACL

5. Yan Xu et al.@2105b **classifying relations via long short term memory networks along shortest dependency paths**ACL
