# Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm
## Abstract 
This paper shows that by extending the distant supervision to a  more diverse set of noisy labels, the 
  models can learn richer representations.
  ## 1. Introduction
  This paper shows that extending the distant supervision to a more set of noisy labels enable the models
  to learn richer representations of emotional content in text.<br>
 **contributions**<br>
  show how millions of emoji occurrence can be used to pretain models to learn a richer emotional
  representation than traditionally obtained through distant supervision.
## 2. Related work
 >method1:<br>prior work has used theories of emotion such as Ekman's six basic emotions and Plutchik's eight basic 
  emotions.<br>
* weakness:<br>requir an understanding of the emotional content of each expresion,which is difficult and 
 time-consuing;prone to misinterpretations and may omit important details.<br>
 >method2:<br>learn emoji embeddings from the words describing the emoji semantics in official emoji tables.<br>
 * limits:<br>①domains with limited or no usage of emojis,②the table do not capture the dynamicx of emoji usage.
  Multitask learning with simultaneous on multiple datasets has shown promissing results. 
 ## 3. Method 
+ pretraining<br>
     this paper use data from Twitter from 2013.1.1 to 2017.6.1, but any dataset with emoji occurrences 
     could be used.
     hypothesis:the content obtained from URL is important for understanding the emotional
     content of the text.
   **Address training data in the following way:**<br>
        for each unique emoji type,this paper save a separate tweet for the pretraining with that emoji type as
        the label,which make the pretraining task a single-label classification.
+ model<br>
**use a variant of the LSTM model.** DeepMoji model uses an embedding layer of 256 dimensions to project each word into 
       a vector space.A hyperbolic tangent activation function is used to enforce a constraint of each embedding dimension
       being with[-1,1].
       the attention mechanism lets the model decide the importance of each word for the prediction task by weighing 
       them when constructing the representation of the text.
       the representation vector for the text is found by a weighted summation over all the time steps useing the attention importance          scores as weight. Adding the attention mechanism and skip-connections improves the model's capabilities for transfer learning.
+ transfer learning<br>
**"chain-thaw",a new simple transfer learning approach**, can sequentially unfreezes and 
       fine-tunes a single layer at a time,which increases accuracy on the target at the expense of extra computational power needed for the fine-tuning.each time the model
      
