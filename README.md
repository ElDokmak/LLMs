![image](https://github.com/ElDokmak/LLMs/assets/85394315/4348f26a-ef5f-446e-9eb6-f2151ee91c69)# Large Language Models (LLMs)
<img src="https://miro.medium.com/v2/resize:fit:1400/1*s2Va5YO3xKPLmrwR2THdtQ.png">

## **Content**
* **What is Large Language Model?**
* **Applications**
* **Types of Language Models**
    * Statistical Modeling
        * N-gram model
        * Exponential Model
    * Neural Modeling
  
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
## **1. Statistical Language Modeling:**

Statistical LM is the development of probabilistic models that predict the next word in a sentece given the words the precede it.

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

     
   
