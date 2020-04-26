paper [link](https://arxiv.org/abs/1909.04164)   	EMNLP 2019
## abstract 
* 从unstructured unlabeled text上训练得到的 contextual word representations不包含任何对真实世界实体的explicit grounding，且经常不能remember facts about those entities。

* KBs可以提供丰富的高质量、人类创造的知识源；包含原始文本中的补充信息；编码factual knowledge，这些知识由于很少提及常识性知识或者long range dependencies，很难从选择偏好中学习。

* 提出一种方法：embed multiple knowledge bases into large scale models，use structured，human-curated knowledge enhance word representations。


**the entity linkers and self-supervised language modeling objective are jointly trained end-to-end in a multitask setting that combines a small amount of raw text(在结合了少量原始文本的多任务设置中进行端到端联合训练)**.
## 1 introduction
large scale pretrained model such as ELMo, GPT, and BERT在a wide range of NLP tasks中significantly improved the state of the art.

* insert multiple KBs into a large pretrained model with a Knowledge Attention and Recontextualization(KAR) mechanism.
> for each KB,first, explicitly model *entity spans* in the input text, use an integrated entity linker to retrieve relevant entity embeddings from a KB **to from knowledge enhanced entity-span representations**;

> then,**recontextualized** the entity-span representations with  **word-to-entity atttention** ，来允许contextual word representations 和上下文中所有实体跨度之间的 long range interactions。

**整个KAR被插入在BERT的中间两层之间(inserted between 2 layers in the middle of BERT)；在未标注数据上学习entity linkers with self-supervision**。benefits of our approach are as follow：
> 1. 没有改变the top layers of the original model，因此在训练KAR时，**可以保留输出损失层，并在未标记语料库上进行微调，这允许在任何下游应用中都可以简单地从BERT切换到KnowBert**。
> 2. 利用原始模型的已有的高容量层，KAR是轻量级的，只添加了最少的额外参数和运行时间。
> 3. 吸收其他额外的KBs十分容易，只需要将他们插入到另外的locations。

使用a mix of intrinsic and extrinsic tasks 来评估KnowBert，外部评估显示在关系抽取，实体类型和词义消岐等任务上，任务表现有提升。
## 2 related work
* 2.1 pretrained word representations: *learning context-sensitive embeddings*

notes of reference paper link [1-EMLo-deep-contextualized-word-representations](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/%E3%80%901802-ELMo%E3%80%91deep-contextualized-word-representations.md) ， [2-BERT-pre-training-of-deep-bidirectional-transformers-for-language-understanding](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/Pre-trainingLM/%40201905_BERT-pre-training_of_deep_bidirectional_Transformers_for_language_understanding.md)

* 2.2 entity embeddings:*从外部知识源 生成 连续向量表示*

基于知识图谱的方法优化知识图谱中观察到的三元组的得分，通过2个主要的策略：**translational distance models which use a distance-based scoring function**和**linear models which use a similarity-based scoring function**

TuckER是啥？

* 2.3 entity-aware language models

adding KBs to generative LMs：[Reference-Aware-Language-Models](https://arxiv.org/pdf/1611.01628.pdf)
> 模型将 reference视为explicit stochasitic latent variable，该构架允许模型通过访问external databases和internal state来创建实体及其属性的mentions。**这有助于将 可在数据库上或语篇上下文中的可预测位置上能访问的信息合并在一起，即使the targets of the reference are rare words。论文的模型变体是基于确定性注意力的**。

building entity-centric LMs：[Dynamic entity representations in neural language models](https://arxiv.org/pdf/1708.00781.pdf)
> 提出一个语言模型EntityNLM：可以显式地建模实体，动态地更新它们的表示，并在上下文中生成它们的mentions。可以在上下文中对任意数量的实体建模，同时以任意长度生成每个实体。

**这种关注实体的语言模型引入了隐变量，这些变量要求训练全标注，或者边缘化**
* 2.4 task-specific KB architecture
intergrate KBs into neural architecture for specific downstream tasks

[Explicit utilization of general knowledge in machine reading comprehension](https://arxiv.org/pdf/1809.03449)
> **MRC模型和人类的阅读理解之间存在的gap表现在：对大量带有answer spans的passage-question pairs的需求 和 对noise的鲁棒性上。MRC models对noisy data十分敏感，鲁棒性差。**。该论文explore how to intergrate the neural networks of MRC models with the general knowledge: **使用WordNet从每一个给定的passage-question pair中抽取inter-word semantic connections 作为general knowledge**；**提出一个end-to-end MRC named as Knowledge Aided Reader，它显性地使用上面抽取到的general knowledge来帮助模型的注意力机制**。

*KBs中存在着 大量以机构化形式存储的general knowledge。常见的KBs有：①WordNet storing semantic knowledge；② ConceptNet storing commonsense knowledge；③ FreeBase storing factoid knowledge*
> **Q:如何理解显性利用抽取到的 general knowledge来帮助模型的attention mechanisims？**

in this paper，the attention mechanisms of KAR(knowledge aided reader) is named as knowledge aided mutual attention and knowledge aided self attention separately.

**the architecture of KAR**

>> **lexicon embedding layer**:*maps the words to the lexicon embeddings,which is composed of its word embedding(obtained via pre-trained GloVe) and character embedding(obtained via CNN operation*;拼接后送入a shared dense layer with ReLU activation;得到$L_{P}$,$L_{Q}$

>> **context embedding layer**:*process the lexicon embeddings with a shared bidirectional LSTM*.得到$C_{P}$,$C_{Q}$

>> **Coarse memory layer** to obtain the question-aware passage representations:使用**knowledge aided mutual attention**将$C_{Q}$融入进$C_{P}$，具体的，需要计算each passage context embedding 和 each question context embedding的similarity。其output表示为$\bar{G}$;使用BiLSTM处理$\bar{G}$，拼接前向LSTM和反向LSTM的outputs得到coarse memories $G\in \mathbb{R}^{d\times n}$

>>**refined memory layer** to obtain final passage representations:使用**knowledge aided self attention**将 G 融入进它本身，其输出表示为$\bar{H}\in \mathbb{R}^{d \times n}$;使用BiLSTM处理$\bar{H}$，拼接前向LSTM和反向LSTM的outputs得到refined memories $H\in \mathbb{R}^{d\times n}$

文中对两种知识辅助注意力都有详细的描述，出于2个原因：不太理解公式中某些变量的具体含义导致没看懂为什么是这样操作 和  跟本文关系不太大， 在此省略。
>> **answer span prediction layer**:预测answer start position 和 answer end position
## 3 knowBert
### 3.1 pretrained BERT
BERT accepts input a sequence of N WordPiece tokens;

computes L layers of D-dimensional contextual representations Hi by applying non-linear functions Hi=Fi(Hi-1), here Fi 是一个multi-headed self-attention layer followed by a position-wise multilayer perception。multi-headed self-attention允许每个vector关注到其他每个vetor。

BERT的训练目标是：最小化$L_{NSP}+L_{MLM}$. MLM使用一个特殊的\[MASK]token 来替代 随机选择的一部分input word pieces，然后在所有可能的word pieces上使用一个linear layer和softmax 来计算the negative log-likelihood of the missing token。
### 3.2 knowledge bases
knowledgs bases adopted in this paper includes KBs with a typical(subj,rel,obj) graph structure,KBs that contain only entity metadata without a graph,and those that combine both a graph and entity metadata.*本文没有假设实体已经定型，这允许 通过使用从实体描述中计算得到的embeddings来直接连接到维基百科页面without a graph*。
> entity candidate selector：take some text as input，return a list of C potential entity links，列表包含Mm个候选实体，每一个都由potential mention span的start 和 end indices组成。

本文选择性地允许 candidate selector 为每一个entity condidate返回 关联先验概率；本文的entity candidate selector 是固定的，但是他们的output被送入一个 可学习的上下文相关的entity linker，来消除candidate mentions的歧义。
### 3.3 KAR-knowledge attention and recontextualizztion
> overflow and 4 key componets:
>> 1. the contextual representations $H_i$ is accepted as input at a particular layer,then projected to $H_{i}^{proj}$;
>> 2. pooled over candidate mentions spans to compute S;
>> 3. contextualized into $S^{e}$ using mention-span self-attention;
>> 4. weighted average entity embeddings $\tilde{E}$ computed by an interated entity linker;
>> 5. $\tilde{E}$ are used to enhance the span representations with knowledge from KB to compute $S'^{e}$
>> 6. recontextualizing the BERT word piece representaitons with word-to-entity-span attention;
>> 7. result of 6 are projected back to the BERT dimension resulting in $H_{i}^{'}$

**4 key componets**: mention-span representations, retrieval of relevant entity embeddings using an entity linker, update of mension-span embeddings with retrieved information, recontextualization of entity-span embeddings with word-to-entity-span attention.
> **key componet 1:mention-span representations**

$H_i$ → projection → $H_{i}^{proj}$ → pooling in a mention-span using self-attentive span pooling   →  stacked into S
$$H_{i}^{proj}=H_{i}W_{1}^{proj}+b_{1}^{proj}$$

> **key componet 2:entity linker**:entity disambiguation for each potential mention from among the available candidates

$S^{e}=TransformerBlock(S)$ using mention-span self-attention,**这允许KnowBert将 global information吸收进每一个linking decision，以便于利用entity-entity cooccurrece，解决几个重叠的candidate mentions中哪一个应该被link的问题**

$S^{e}$被用于 当从KB中吸收candidate entity prior时，为每一个condidate entity进行打分。**每一个candidate span $m$ 都有 ①an associated mention-span vetor $s_{m}^{e}$ , ②Mm个带embedding $e_{mk}$的candidate entities , ③ 先验概率 $p_{mk}$** ；我们对 （先验、entity-span vectors和entity embeddings之间的点乘） 使用a 2-layer MLP 来计算出Mm个分数：$$\varphi \_{mk}=MLP(p_{mk},s_{m}^{e}\cdot e_{mk})$$
> **key componet 3:knowledge enhanced entity-span representations**:inject KB entity information into the mention-span representations to form entity-span representations

对于每一个给定的span m，首先忽略那些在 fixed threshold 之下的 candidate entities，并使用softmax对剩下的scores正则化得到$\tilde{\varphi \_{mk}}$;

得到weighted entity embedding: 
$$ \tilde{e_{m}}=\sum_{k}\tilde{\varphi \_{mk}}e_{mk}$$
如果所有的entity linking scores 都在threshold 之下，我们将使用一个特殊的NULL来代替$ \tilde{e_{m}}$

使用weighted entity embeddings 来更新entity-span representations：
$$s_{m}^{'e}=s_{m}^{e}+\tilde{e}\_{m}$$

$s_{m}^{'e}$被打包进a matrix $S'^{e}\in \mathbb{R}^{C\times E}$
> **key componet 4:recontextualization with word-to-entity-span attention**
When recontextualizing the word piece representations, we *use a modified transformer layer that **substitutes the multi-headed self-attention with a multi-headed attention between the projected word piece representations and knowledge enhanced entity-span vectors***.,which means use $H_{i}^{proj}$ for the query, $S'^{e}$ for both the key and value.
$$H_{i}^{'proj}=MLP(MultiHeadAtt(H_{i}^{proj},S'^{e},S'^{e}))$$
这允许each word piece 可以关注到 上下文中的所有entity-spans，使得模型可以在long contexts上传播entity information。
> **alignment of BERT and entity vectors**

由于KnowBert没有对entity embeddings做任何限制，因此十分有必要将entity embeddings 与 pretrained BERT contextual representations对齐，于是，我们将 $W_{2}^{proj}$初始化为 $W_{1}^{proj}$的逆矩阵。
### 3.4 training procedure

## 4 experiments

### 4.1 setup
### 4.2 intrinsic evauation
### 4.3 downstream tasks
## 5 conclusion

