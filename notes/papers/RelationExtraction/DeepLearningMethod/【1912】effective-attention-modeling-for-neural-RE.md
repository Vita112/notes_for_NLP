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
使用a linear form of attention来找到 句子中关于entities的 有语义意义的words，这些能够为实体间的关系提供证据。

使用entities作为attention queries；命名实体大多数由multiple tokens组成；一个实体的nearby words可以提供关于这个实体的有用信息。**本文使用the tokens of an entity and its nearby tokens 来得到entity representation vector，使用CNN with max-pooling in the context of an entity来获得其表示**。具体的，cnn网络层的输入不是一个完整的自然句，而是句子中包含an entity and its neighboring context的sequence，分别经过 $f_{e}$个filters得到2个实体的向量表示$v_{e}^{1}$和$v_{e}^{2}$。

ues a linear function to measure the semantic similarity of words with the given entities：
$$f_{score}^{1}(h_{i},v_{e}^{1})=h_{i}^{T}W_{a}^{1}v_{e}^{1}$$
其中，$W_{a}^{1}$是可训练的权重向量，$$f_{score}^{1}(h_{i},v_{e}^{1})$表示第i个单词和实体$v_{e}^{1}$的语义相似度。


由于靠近实体的单词，对于找到2个实体间的关系更为重要，因此**本文提出incorporate the syntactic structure of a sentence in the attention mechanism**:
> 句法结构通过句子的dependency parse tree得到，我们将 从实体的head token(last token)到每个单词的dependency distance 定义为沿着依赖路径的边数。我们使用一个距离窗口大小ws，且依存距离在这个窗口大小内的words will receive attention，其他的单词将被忽略。我们遮蔽了那些 距离2个实体的平均依存距离大于ws的words.*此时，得到了在特定窗口内进行attention的 包含句子依存距离信息的attention scores*,然后进行normalization。

> 同时使用words的词义和他们距离2个实体的依存距离，结合上一步得到的normalized attention score，得到了**the attention feature vector $v_{a}^{1}$, $v_{a}^{1}$ with respect to the two entities**:
$$v_{a}^{1}=\sum_{i=1}^{n}p_{i}^{1}h_{i}$$

### 3.3 multi-factor attention
the number of factors 是一个hyper-parameter，the model replace the attention matrix $W_{a}$ with an attention tensor $W_{a}^{1:m}$,此处m是factor count,$W_{a}^{1:m} \in \mathbb{R}^{m\times 2(d_{w}+d_{z})\times 2f_{e}}$.然后，得到关于每个实体的m个attention vectors，拼接所有特征向量得到multi-attentive feature vector $v_{ma}\in \mathbb{R}^{4m(d_{w}+d_{z})}$.

### 3.4 relation extraction

拼接$v_{g},v_{ma},v_{e}^{1},v_{e}^{2}$后，将其送入 a feed-forward layer with softmax activation，来预测relation labels的normalied probabilities。
$$r= softmax(W_{r}(v_{g}||v_{ma}||v_{e}^{1}||v_{e}^{2})+b_{r})$$

### 3.5 loss function
use negative log-likelihood as our objective function:
$$L=-\frac{1}{B}\sum_{i=1}^{B}log(p(r_{i}|s_{i},e_{i}^{1},e_{i}^{2},\theta ))$$
## 4 experiments
### 4.1 datasets
New York Times (NYT) corpus with 2 versions:
> The original NYT corpus created by Riedel et al. (2010) which has 52 valid relations and a None relation. We name this dataset NYT10. 

>Another version
created by Hoffmann et al. (2011) which has 24 valid relations and a None relation. We name this dataset NYT11. 
### 4.2 evaluation metrics

### 4.3 parameter settings

### 4.4 comparison to prior work
(1) CNN (Zeng et al., 2014): 

(2) PCNN (Zeng et al., 2015):

(3) Entity Attention (EA) (Shen and Huang,2016): 

a model which is a combination of a CNN with max-pooling and an attention mechanism. CNN with max-pooling 被用于抽取global features；

每个word的tvector representation 将和 last token of the entity 的word embedding进行拼接。

拼接后的representation 被送入a feed-forward layer with tanh activation，以及另一个feed-forward layer来获得 a scalar attention score for every word。原始的word representations 将基于attention scores进行平均得到attentive feature vectors.

基于CNN抽取的全局特征和 2个关于2个实体的attentive feature vectors进行拼接后，送入一个带有softmax的前向反馈层，来决定relation。

(4) BiGRU Word Attention (BGWA) (Jat et al.,2017):

(5) BiLSTM-CNN: baseline

### 4.5 experimental results

## 5 analysis and discussion
### 5.1 varying the number of factors

### 5.2 effectiveness of model components

### 5.3 performance with varying sentence length and varying entity pair distance
随着句子长度变长，以及2个实体间距离越来越远，所有模型的表现都下降了。与其他模型相比，本文的multi-factor attention model with dependency-distance weight factor 提高了F1 score。

## 6 related work
*　关于intra-sentence 和 cross-sentece
Surdeanu et al. (2012), Lin et al.(2016), Vashishth et al. (2018), Wu et al. (2019),and Ye and Ling (2019) used multiple sentences in a **multi-instance relation extraction** setting to *capture the features located in multiple sentences for a pair of entities*. 他们将拥有相同实体对的多个句子作为一个测试实例来评估模型表现。

本文独立地在每个句子上进行训练，在sentence level上评估。**未来的工作将考虑将本文的模型扩展至multi-instance relation extraction framework中**。

* 关于结合一个句子的 dependency structure info 用于RE

* attention-based NN的应用： NMT,answer span extraction等
