paper [link](https://arxiv.org/pdf/1905.08284.pdf)  written by staffs from 阿里巴巴美国
* abstract
RC任务依赖于来自 the sentence 和 the 2 target entities两方面的信息。论文提出一种模型：利用BERT得到的句子表示，并吸收2个目标实体的信息，用于关系分类任务。具体地，使用special token定位target entities，通过pre-trained BERT to transfer information，incorporate the embeddings of target entities to tackle RC task。
* main contributions：

> text被送入BERT进行微调之前，分别在e1和e2的input representations的前后各插入a pecial token(也就说，总共有4个special tokens).这样做的目的是：**定位2个目标实体的位置，并将信息传递到BERT中**。

> 拼接 sentence embedding from BERT and embeddings of target entities后，将其作为input送入a multi-layer neural network，用于关系分类。

* methodology：model architecture

在插入\[CLS]和special tokens之后，一个句子表示为：*“\[cls] The $kichen$ is the last renovated part of the # house #.”* 

**\[CLS]对应的BERT输出向量为整篇文档的语义表示，它融合了句子中每个字\词的semantic information**

由于句子在被送入BERT之前，分别在2个目标实体的前后插入了special tokens，因此，句子的BERT 表示输出中可以很容易定位2个目标实体的始终表示位置，*在得到的entity-span representation上apply **average operation** to get a vector representation；then employ an **activation operation（tanh）**；then add a fully connected layer to the entity-span embedding*。这样便得到了 2个目标实体的embeddings；

拼接 \[CLS] embedding and embeddings of target entities后,将其送入一个fully-connected layer，再对得到的输出做softmax 操作得到最终的分类结果。

* ablation studies

分析结果显示：the special separate tokens can identify the locations of the 2 target entities and transfer the information into the BERT model。

* future work： to extend the model to apply to distant supervision。
