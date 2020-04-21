[paper link](https://arxiv.org/pdf/1912.03832.pdf)

#### comments:Accepted at CoNLL 2019

## abstract
* 问题点①  句子长，句子中两个实体相隔较远；2个实体的链接是indirect，比如connected via a third entity or via coreference

**solution：使用dependency parser来获取句子的句法结构；使用a linear form of attention 来衡量单词与给定实体的semantic similarities，并将它与单词距离给定实体的dependency distance结合起来，用于衡量它们对识别关系的影响。**
* 问题点②  the words in a sentence don't contribute equally in identifying the relation between two entities。

**solution：使用multi-factor attention**
## introduction
远程监督方法缓解了标注数a据不足的问题，**但同时也产生了大量noisy training instances。**
* feature-based learning models：lexical，syntactic等特征
* network-based models：CNN with max-pooling 
> Limmitations: 在理解单词和给定实体的语义相似性上存在限制；不能捕获long-distance dependencies，比如单词和共指实体之间的关系
## task description
本文解决sentence-level RE。任务定义为：给定一个句子S和句子中已标注的2个实体，从一个pre-defined关系集R∪﹛None﹜中找到这对实体间的关系，其中None代表 给定句子中标注的实体对之间，不存在 属于R的任何一种关系。*实体间的关系是 argument order-specific，例如r(E1，E2)≠r(E2，E1)*
## Model description
* 4种embedding vectors
> * word embedding vector **W**
> * entity token indicator embedding vector **Z**：表明一个单词是属于E1，还是E2，或者不属于任何实体
> * positional embedding vector **u1** ：表示一个单词 与E1的start token 的线性距离
> * positional embedding vector **u2**：表示一个单词 与E2的start token 的线性距离

* use Bi-LSTM layer to capture interaction among words in a sentences，本层的输入是 **W**和**Z**的拼接，得到t-th时间步的Bi-LSTM的输出$h_{t}$.
### 3.1 global feature extraction
使用CNN来抽取sentence-level global features：拼接**u1**，**u2**和$h_{t}$；use CNN with max-pooling on concatenated vectors 来抽取全局特征向量$v_{g}$, 即每经过一个convolutional filtr evector $f_{i}$和max-pooling operation后将得到一个scale value $c_{max}^{i}$,我们有$f_{g}$个filter，最终得到a global feature vector
$$v_{g}=\[c_{max}^{1},c_{max}^{2},\cdots ,c_{max}^{f^{g}}]$$
### 3.2 attention modeling
 
