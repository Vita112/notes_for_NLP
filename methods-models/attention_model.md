## 1 人类的视觉注意力
借鉴 人类的注意力机制。先从视觉的选择性注意力机制出发：
![视觉注意力机制]()

上图显示了人类视觉接收一幅图时，如何高效分配有限的注意力资源；红色区域表明视觉系统更关注的目标，也可看做是**注意力的焦点**，
对这一区域投入更多的注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息。
## 2 encoder-decoder 框架
目前大多数注意力模型附着在 encoder-decoder框架下，该框架是一种深度学习领域的研究模式。

在文本处理领域，endoder-decoder框架可以直观理解如下：将其看做适合处理一个句子（或篇章）生成另一个句子（篇章）的通用处理模型。
>对于句子（source，target），目标是给定输入句子source，期待通过encoder-decoder框架来生成目标句子target，source和target
可是同一种语言，也可是不同语言。

+ encoder：对输入句子source进行编码，将输入句子通过非线性变换转化为中间语义表示C：
$$\mathbf{C}=\mathbf{F}(x_{1},x_{2},\cdots ,x\_{m})$$

+ decoder：根据句子source的中间语义表示C和之前生成的历史信息
$$y_{1},y_{2},\cdots ,y_{i-1}$$
来生成i时刻生成的单词$y_{i}$,所以
$$y_{i}=\mathbf{G}(\mathbf{C},y_{1},y_{2},\cdots ,y\_{i-1})$$
+ 应用
> 1. source:中文句子；target：英文句子。   →    机器翻译
> 2. source:一篇文章；target：该文章的概括性描述。   →   文本摘要
> 3. source：图片；target：一句描述语，可以描述图片的语义内容
## 3 Attention model
本节内容：首先 以MT为例讲解soft attention模型基本原理；然后 抽象出注意力机制的本质思想； 最后 简单介绍self attention。
### 3.1 soft attention 
观察encoder-decoder框架，可以发现：**encoder-decoder框架并没有体现出*注意力模型*，因为decoder非线性函数在生成目标句子的单词时，
不论生成哪个单词，所使用的输入句子source的语义编码C都是一样的，这意味着对于生成某个目标单词yi来说，句子source中的任意单词具有
相同的影响力，*但实际上，在decoder阶段，输入句子source的语义编码C中的每个单词对于生成某个目标单词yi的影响力是不同的*，因此这个
框架中并没有引入注意力机制**。
> 为何要引入注意力机制？
> 未引入attention 的encoder-decoder框架，在处理较短的输入句子时问题不大，**但如果输入句子较长，此时，所有的
语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，这会丢失很多细节信息**。

+ 引入attention机制

引入attention 后，在编码句子语义阶段，针对source每个单词，注意力模型会给不同的单词分配注意力大小，即**原先固定的中间语义
表示C 被替换成 根据当前输出单词来调整加入注意力模型的变化的Ci**,每个Ci对应着不同源语句子单词的注意力分配概率分布。框架图如下：
![引入注意力机制的edcoder-decoder框架]()

+ 以中英机器翻译例句“Tom chase Jerry”为例，说明attention-based source单词的中间语义表示：

句子语义编码表征为（因latex公式中似乎不识别中文，故用拼音代替）：
> \mathbf{C}\_{tangmu}=g(0.6f_{2}("Tom"),0.2f_{2}("Chase"),0.2f_{2}("Jerry"))

> \mathbf{C}\_{zhuizhu}=g(0.2f_{2}("Tom"),0.7f_{2}("Chase"),0.1f_{2}("Jerry"))

> \mathbf{C}\_{jierui}=g(0.3f_{2}("Tom"),0.2f_{2}("Chase"),0.5f_{2}("Jerry"))

**f2变换函数 表示encoder对输入英文单词进行某种变换**，若encoder使用RNN模型，则f2函数输出的结果是 某个时刻输入xi后隐层节点的状态值；
**g变换函数 表示encoder根据单词的中间表示 合成整个句子中间语义表示**，故加入attention机制后，source输入句子的单词语义表示Ci本质上
是一个加权求和函数：
$$\mathbf{C}\_{i}=\sum_{j=1}^{L_{x}}\alpha \_{ij}h_{j}$$

$L_{x}$表示 输入句子source的长度；$\alpha \_{ij}$表示 在target输出第i个单词时，source输入句子中第j个单词的注意力分配系数，hj表示 source输入句子中第j个单词的语义编码.下图更为直观：
![attention-based_source单词语义表示Ci形成过程]()

+ 如何获取source输入句子中单词的注意力分配概率值分布？
> RNN作为具体模型的encoder-decoder框架

![RNN-based_encoder-decoder框架下注意力分配概率计算]()
**我们的目标：计算输出yi时，输入句子source中的每个单词xi对应于yi的注意力分配概率分布，也可以理解为某个输入句子单词xi和目标生成单词的对其概率**。于是，我们*通过将target输出句子i-1时刻的隐层节点状态$H_{i-1}$ 和 输入句子source中每个单词对应的RNN隐层节点状态hj 进行对比，即使用函数F(hj,Hi-1)来获得目标单词yi和每个输入单词对应的对齐可能性*。F函数可采取不同的方法。*使用softmax函数 对F函数的输出进行归一化处理，得到符合概率分布取值区间的注意力分配概率分布数值*。
### 3.2 essence of Attention medel

### 3.3 self attention
## 4 Application of Attention model
[Reference blog1](https://zhuanlan.zhihu.com/p/37601161)

[Reference blog2](https://blog.csdn.net/mpk_no1/article/details/72862348)
