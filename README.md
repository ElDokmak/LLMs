# Large Language Models (LLMs)
<img src="https://www.marktechpost.com/wp-content/uploads/2023/04/Screenshot-2023-05-01-at-6.33.37-AM.png">

## **Content**
* **What is Large Language Model?**
* **Applications**
* **Types of Language Models**
    * Statistical Modeling
        * N-gram model
        * Exponential Model
    * Neural Modeling
* **Evaluation of LMs**
* **Transformer-Based Language Models**
  
---
## What is Large Language Model?
Language model in simple words is a type of Machine Learning models that can perform a variety of NLP tasks such as generating and classifying text, translation and Q&A.
With a main goal which is to assign probablities to a sentece.

The term **large** refers to the number of parameters that model has.

LLMs are trained with huge amount of data and use self-supervised learning to predict the next token in a sentence, given the surrounding context,
this process is repeated until the model reaches an acceptable level of accuracy.

---
## Applications
1. Text Generation
2. Machine Translation
3. Summarization
4. Q&A, dialogue bots, etc.....

---
## Types of Language Models?
## **Statistical Language Modeling:**
> Statistical LM is the development of probabilistic models that predict the next word in a sentece given the words the precede it.

### N-gram Language Models: 
N-gram can be defined as the contigous sequence of n items (can be letters, words or base pairs according to the application) from a given sample of text (A long text dataset).

An n-gram model predicts the probability of a given n-gram within any sequence of words in the language.

A good n-gram model can predict the next word in the sentence i.e the value of **p(w|h)**
where w is the word and h is the history or the previos words.
```
This repo is about LLMs
Unigram ("This", "repo", "is", "about", "LLMs")
bigrams ("This repo", "repo is", "is about", "about LLMs")
```
You can think of n-gram model as counting frequncies as follows: consider the previos example 
```
P(LLMs|This repo is about) = count(This repo is about LLMs) / count(This repo is about)
```
Now to find the next word in a sentence we need to calculate the **p(w|h)**, let's consider the above example 
```
P(LLMs|This repo is about)
After generalization: P(wn|w1,w2,.....,wn-1)
```
But how we calculate it?
```
P(A|B) = P(A,B)/P(B)
P(A,B) = P(A|B)P(B)
After generalization:
P(x1,x2,x3,....,xn) = P(x1)P(x2|x1)P(x3/x1,x2)......P(xn|x1,x2,....,xn-1) # Chain Rule
P(w1w2w3....wn) = ΠP(wi|w1w2....wn)
Simplifying using Markov assumption:
P(wi|w1,w2,...,wi-1) = P(wi|wi-k,....,wi-1)
For Unigram: P(w1w2,....wn) = ΠP(wi)
For Bigram: P(wi|w1w2,....wi-1) = P(wi|wi-1)
```
We can extend to trigrams, 4-grams, 5-grams.

> In general this is an insufficient model of language because language has long distance dependecies:
> The computer(s) which I had just put into the machine room on the fifth floor is (are) crashing.
     
### Exponential Language Models: 
Exponential (Maximum entropy) LM encode the relationship between a word and the n-gram history using feature functions. The equation is 

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/99beafb1eed251682c4037f19a4d80da3003cd4e">
Where Z(w1,....,wm-1) is the partition function, a is the parameter vector, and f(w1,......,wm) is the feature function.
In the simplest case, the feature function is just an indicator of the presence of a certain n-gram. It is helpful to use a prior on 
a or some form of regularization.

The **log-bilinear model** is another example of an exponential language model.

### Limitations of Statistical LMs
* Sparsity probelm
     1. Count(n-gram) = 0 --> Solution: Smoothing (adding small alpha)
     2. Count(n-1 gram) = 0 --> Solution: Backoff
* Exponential growth
     The number of n-grams grows as an nth exponent of the vocabulary size. A 10,000-word vocabulary would have 10¹² tri-grams and a 100,000 word vocabulary will have 10¹⁵ trigrams.
* Generalization
     Lack of generalization. If the model sees the term ‘white horse’ in the training data but does not see ‘black horse’, the MLE will assign zero probability to ‘black horse’. (Thankfully, it will assign zero probability to Purple horse as well)
  
## **Neural Language Modeling: (NLM)**
NLM mainly use RNNs (/LSTMs/GRUs)

### RNN’s (Recurrent Neural Networks):
> A recurrent neural network (RNN) processes sequences by iterating through the sequence elements and maintaining a state containing information relative to what it has seen so far. In effect, an RNN is a type of neural network that has an internal loop.

RNNs solve the sparsity problem
<img src="https://stanford.edu/~shervine/teaching/cs-230/illustrations/architecture-rnn-ltr.png?9ea4417fc145b9346a3e288801dbdfdc">

**Types of RNNs**
<img src="https://miro.medium.com/v2/resize:fit:1400/1*VCxoP7Siu0YB501L_-eg1w.png">

**Disadvantages of RNNs**
* Vanishing Gradiients --> Solution: LSTMs/GRUs.
* Exploding Gradients --> Solution: Truncation or squashing the gradients.
* Sequential processing/ slow.
* Can't capture information for longer sequences.

### LSTMs (Long Short Term Memory networks)
> LSTMs are a special kind of RNN, capable of learning long-term dependencies.
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2017/12/10131302/13.png">

LSTM is made up of 3 gates.
1. Input Gate: We decide to add new stuff from the present input to our present cell state scaled by how much we wish to add them.
2. Forget Gate: After getting the output of previous state, h(t-1), Forget gate helps us to take decisions about what must be removed from h(t-1) state and thus keeping only relevant stuff.
3. Output Gate: Finally we’ll decide what to output from our cell state which will be done by our sigmoid function.

---
## Evaluation of LMs
### Extrinsic Evaluation
> Extrinsic evaluation is the best way to evaluate the performance of a language model by embedding it in an application and measuring how much the application improves.

End-to-end evaluation where we can understand if a particular improvement in a component is really going to help the task at hand, however it's time consuming

### Intrinsic Evaluation
> An intrinsic evaluation metric is one that measures the quality of a model-independent of any application.
* Perplexity: It is a measure of how well a probability model predicts a sample.
Perplexity is the inverse of probability and lower perplexity is better



