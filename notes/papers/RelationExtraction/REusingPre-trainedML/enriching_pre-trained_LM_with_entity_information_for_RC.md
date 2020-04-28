paper [link](https://arxiv.org/pdf/1905.08284.pdf)  written by staffs from 阿里巴巴美国
## abstract
RC任务依赖于来自 the sentence 和 the 2 target entities两方面的信息。论文提出一种模型：利用BERT得到的句子表示，并吸收2个目标实体的信息，用于关系分类任务。具体地，使用special token定位target entities，通过pre-trained BERT to transfer information，incorporate the embeddings of target entities to tackle RC task
* main contributions：

> text被送入BERT进行微调之前，分别在e1和e2的input representations的前后各插入a pecial token(也就说，总共有4个special tokens).这样做的目的是：**定位2个目标实体的位置，并将信息传递到BERT中**。
