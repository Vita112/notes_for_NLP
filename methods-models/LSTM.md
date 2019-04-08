## 1 初识LSTM
![LSTM(1)](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/LSTM_model(1).jpg)

![LSTM(2)](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/LSTM_model(2).jpg)

## 2 RNN 与 LSTM 
LSTM 是RNN的一种特殊的RNN类型，它通过在循环单元内的多层操作，使得网络能够处理 long-term dependency on sentences，解决了RNN
在处理文本序列的长期依赖信息时表现不好的问题(表现不好具体指：传统的RNN在进行几次链式法则求导后，梯度会指数级缩小，导致传播几层后，
出现梯度消失，因而无法处理长期依赖问题)。

RNN循环单元内部(隐层)的操作比较简单，通常是一个tanh 函数；LSTM同样可以看作是同一神经网络的多次复制，与RNN相比不同的是：
LSTM的隐层操作更加复杂，正如上图所示，通过3个门对输入进行控制，以便得到我们期望的结果。门可以选择性的控制信息的流动，通常由一个sigmoid神经网络和一个point wise(element wise)的乘法操作组成。下图显示了RNN与LSTM隐层操作的不同：
![difference_between_RNN_and_LSTM]()

通俗来讲，LSTM中，循环的神经网络通过使用3个门，来控制处理序列的信息，模型可以有选择性的保留序列的特征，留下重要的部分，遗忘掉不重要的部分。
在forget gate中，通过使用sigmoid function对上一隐层状态$h_{t-1}$和输入向量$x_{t}$进行操作，输出值映射在\[0, 1]的范围内，这个值反映了
网络将遗忘之前信息的哪些部分。
## 3 LSTM 的一些变体
+ 增加peephole connections(窥视孔连接)

[Gers & Schmidhuber (2000)]()提出的增加peephole connections,如下图：
![LSTM_with_peephole_connections]()

如图，在所有的门之前都与上一时刻t-1状态线相连，使得状态信息对门的输出值产生影响。使用时，可能并不是在每个门中都加入$C_{t-1}$.
+ 耦合遗忘门和输入们

将遗忘门和输入门耦合在一起，也就是说 **遗忘多少就更新多少新状态，没有遗忘就不更新状态，全部遗忘就把新状态全部更新进去**。
![coupled_forget_gate_and_input_gate]()
+ GRU

![GRU]()
将遗忘门和输入门统一为更新门，且合并h和c。可参考[可参考Cho, et al. (2014)]()
## reference
1. [bolg 1 from CSDN-LSTM中几个门的理解](https://blog.csdn.net/zhuiqiuzhuoyue583/article/details/80381041)

2. [bolg 2 from CSDN-【译文】理解LSTM网络](https://www.jianshu.com/p/9dc9f41f0b29)
