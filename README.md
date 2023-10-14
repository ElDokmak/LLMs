# Large Language Models (LLMs)
<img src="https://www.marktechpost.com/wp-content/uploads/2023/04/Screenshot-2023-05-01-at-6.33.37-AM.png">



---
## **Content**
* **[What is Large Language Model?](#what-is-large-language-model)**
* **[Applications](#Applications)**
* **[Types of Language Models](#types-of-language-models)**
    * [Statistical Modeling](#statistical-language-modeling)
        * [N-gram model](#N-gram-Language-Models)
        * [Exponential Model](#Exponential-Language-Models)
    * [Neural Modeling](#neural-language-modeling-nlm)
* **[Evaluation of LMs](#evaluation-of-lms)**
* **[Transformer-Based Language Models](#transformer-based-language-models)**
* **[Refrences](#Refrences)**
* **[Generative configuration parameters for inference](#generative-configuration-parameters-for-inference)**
* **[Computational challenges and Qunatization](#Computational-challenges-and-Qunatization)**
* **[Prompt Engineering](#prompt-engineering)**
* **[Fine-Tuning](#fine-tuning)**


  
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
* Sparsity probelm:
     1. Count(n-gram) = 0 --> Solution: Smoothing (adding small alpha)
     2. Count(n-1 gram) = 0 --> Solution: Backoff
* Exponential growth:
  
     The number of n-grams grows as an nth exponent of the vocabulary size. A 10,000-word vocabulary would have 10¹² tri-grams and a 100,000 word vocabulary will have 10¹⁵ trigrams.
* Generalization:
  
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



---
## Transformer-Based Language Models
Previosly we discussed some of the RNNs (LSTMs) Challenges: like slow computations due to sequential processing, and they can't capture contextual information for longer sequences.

> The Solution was found when the paper Attention is All You Need which introduced the Transformer architecture came to life.

> Transformers has two main components: Encoder and Decoder (explained with code [here](https://github.com/ElDokmak/LLMs/tree/main/Transformer-Based%20Language%20Models))


### Encoder: 
It is a bidirectional transformer network that encodes inputs, it takes in text, produces a feature vector representation for each word in the sentence.
> Encoder Models: Masked Language Modeling (auto-encoders) like BERT, ALBERT, ....etc.
>
> Use cases: Sentiment analysis, NER, and Word classification


### Decoder: 
It is a uni-directional transformer network that generates output.
> Decoder Models: Causal Language Modeling (Autoregressive) like GPT, BLOOM.
>
> Use cases: Text generation.


### Encoder + Decoder Models:
> Span corruption modes like T5, BART.
>
> Use cases: Translation, Text summarization, and Question answering.



---
## Generative configuration parameters for inference
<img src="https://github.com/ElDokmak/LLMs-variety/assets/85394315/b8f7cc45-8b97-411f-82df-711988a90b50">

* **Max new tokens:** the number of generated tokens.
* **Greedy vs. random sampling:**
   * **Greedy search:** the word/token with the highest probability is selected.
   * **Random sampling:** select a token using a random-weighted strategy across the probabilities of all tokens.
<img src="https://miro.medium.com/v2/resize:fit:1400/1*3WS82V-mcwbiGvNupuWoCw.jpeg">

* **Top-k sampling:** select an output of the top-k results after applying random-weighted strategy using the probabilities.
* **Top-p sampling:** select an output using the random-weighted strategy with the top-ranked consecutive results by probability and cumulative probability <= p. 
* **Temperature:** you can think of temperature as how creative out model is?
   * **Lowwer-temp:** means strongly peaked probability distribution with more certain and realistic outputs.
   * **Higher-temp:** means broader, flatter probability distribution with creative output but less certain.



---
## Computational challenges and Qunatization
> Approximate GPU RAM needed to store/train 1B parameters

> To store 1 parameter = 4 bytes(FP32) --> 1B parameters = 4GB 
>
> To train 1 parameter = 24 bytes(FP32) --> 1B paramters = 80GB

This huge usage of GPU RAM will result in Out Of Memory problem: as a solution **Quantization** was introduced

## Quantization
Instead of full precision we can use lower precision.

The following image shows how to store paramters using different data types:
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*VAq-r_4DnhmZcsaI4rqw3A.png">

### Types of Quantization
1. **Post-Training Quantization (PTQ)**
   > Is a straightforward technique where the weights of an already trained model are converted to lower precision without necessitating any retraining.
   > Although easy to implement, PTQ is associated with potential performance degradation.
   
   > We will focus on PTQ only.
2. **Quantization-Aware Training (QAT)**
   > incorporates the weight conversion process during the pre-training or fine-tuning stage, resulting in enhanced model performance. However, QAT is computationally expensive and demands representative training data.


<!--
### 🔰 Naïve 8-bit Quantization
* A symmetric one with **absolute maximum (absmax) quantization**.
<img align="center" src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*jNl_x4JF0lpRA4_6Cae9cg.png">

```
import torch

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant
```

* An asymmetric one with **zero-point quantization**.
<img align="center" src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*GiCuvGWBtdU4-hXcvetTnw.png">

Then, we can use these variables to quantize or dequantize our weights:
<img align="center" src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*SalTtt_eNoYOeHLD1XiEfw.png">

```
import torch

def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale = 255 / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant
```

> In both cases the goal is to map FP32 into INT8.

> e.g. We have a maximum value of 3.2 and a minimum value of -3.0. We can calculate the scale is 255/(3.2 + 3.0) = 41.13 and the zero-point -round(41.13 × -3.0) - 128 = 123 -128 = -5, so our previous weight of 0.1 would be quantized to round(41.13 × 0.1 -5) = -1.
> This is very different from the previous value obtained using absmax (4 vs. -1).
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*n5nqoJUXp65JahKsLzQS-A.png">

-->

### GPTQ, AutoGPTQ, GGML, Pruning, and Distillation
> Explained in the directory **[Quantization](https://github.com/ElDokmak/LLMs/tree/main/Quantization)**



---
## Prompt Engineering
### **In-context learning (ICL)**
> In-context learning (ICL) is a specific method of prompt engineering where demonstrations of the task are provided to the model as part of the prompt (in natural language).

> LLMs demonstrate an in-context learning (ICL) ability, that is, learning from a few examples in the context. Many studies have shown that LLMs can perform a series of complex tasks through ICL, such as solving mathematical reasoning problems.

* **zero shot inference**
```
Classify this review:
I love chelsea
Sentiment:

Answer: The sentiment of the review "I love Chelsea" is positive.
```

* **one/few shot inference**
```
Classify this review:
I love chelsea.
Sentiment: Positive

Classify this review:
I don't like Tottenham.
Sentiment:

Answer: The sentiment of the review "I don't like Tottenham" is negative.
```
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each van has 3 balls. How many tennis balls does he have now?
A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: The answer is 27.
```

> It is observed that standard prompting techniques (also known as general input-output prompting) do not perform well on complex reasoning tasks, such as arithmetic reasoning, commonsense reasoning, and symbolic reasoning.

* **Chain-of-Thought Prompting**
> CoT is an improved prompting strategy to boost the performance of LLMs such non-trivial cases involving reasoning.
>
> CoT incorporates intermediate reasoning steps that can lead to the final output into the prompts.
<!--
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each van has 3 balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5+6=11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23-20=3. They bought 6 more apples, so they have 3+6=9. The answer is 9.
```
-->
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*D4AEQft-b2VmXb07dIVONg.png">

* **Zero-shot CoT**
> In Zero-shot CoT, LLM is first prompted by “Let’s think step by step” to generate reasoning steps and then prompted by “Therefore, the answer is” to derive the final answer.
>
> They find that such a strategy drastically boosts the performance when the model scale exceeds a certain size, but is not effective with small-scale models, showing a significant pattern of emergent abilities.
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*l7oj25kVU6ALI5IbBb2BUA.png">

> While Zero-shot-CoT is conceptually simple, it uses prompting twice to extract both reasoning and answer, as explained in the figure below.
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*VvezFJ4L5Ur1he-qT8scAg.png">

> The process involves two steps: first “reasoning prompt extraction” to extract a full reasoning path from a language model, and then use the second “answer prompt extraction” to extract the answer in the correct format from the reasoning text.

* **Self-consistency COT**
> Instead of using the greedy decoding strategy in COT, the authors propose another decoding strategy called self-consistency to replace the greedy decoding strategy used in chain-of-thought prompting.

> First, prompt the language model with chain-of-thought prompting, then instead of greedily decoding the optimal reasoning path, authors propose “sample-and-marginalize” decoding procedure.
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*aHxtX5BCVcJ5rHaqB1aHaw.png">

* **Tree of thoughts**
> Tree of thoughts which generalizes over the “Chain of Thoughts” approach to prompting language models and enables exploration over coherent units of text (“thoughts”) that serve as intermediate steps toward problem-solving.

> ToT allows LMs to perform deliberate decision-making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices.

> The results/experiments show that ToT significantly enhances language models’ problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords.
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*QJidAN1BWinejmLF8ByRqw.png">

The ToT does 4 things:
   1. Thought decomposition
   2. Thought generator
   3. State evaluator
   4. Search algorithm.

* **Self-Ask**
> Self-Ask Prompting is a progression from Chain Of Thought Prompting, which improves the ability of language models to answer complex questions.
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*5R8lx3gTlEjvvHoSs0Gjhg.png">

### **Auto Prompt Techniques**
> This is an active research area and the following section discusses some attempts towards automatic prompt design approaches.

**1. Automatic Prompt Augmentation and Selection COT:**

Prompt Augmentation and Selection COT is a three step process:
- **Augment:** Generate multiple pseudo-chains of thought given question using few-shot or zero-shot CoT prompts;
- **Prune:** Prune pseudo chains based on whether generated answers match ground truths.
- **Select:** Apply a variance-reduced policy gradient strategy to learn the probability distribution over selected examples, while considering the probability distribution over examples as policy and the validation set accuracy as reward.

**2. Auto-CoT: Automatic Chain-of-Thought Prompting:**

In Automatic Chain-of-Thought Prompting in Large Language Models, the authors propose Auto-CoT paradigm to automatically construct demonstrations with questions and reasoning chains.
In this technique, authors adopted clustering techniques to sample questions and then generates chains.

<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*-3I3jfHy-_6vYbViWM1lKw.png">

Auto-CoT consists of the following main stages:
- **Question clustering:** Perform cluster analysis for a given set of questions Q. First compute a vector representation for each question in Q by Sentence-BERT.
- **Demonstration selection:** Select a set of representative questions from each cluster; i.e. one demonstration from one cluster.
- **Rationale generation:** Use zero-shot CoT to generate reasoning chains for selected questions and construct few-shot prompt to run inference.




---
## Fine-Tuning
1. In-context learning
2. Feature Based Finetuning
3. PEFT (Parameter Efficient Fine Tuning)
4. RLHF (will be updated later)



---
## Refrences        
|                                       Refrences                                    |
| :--------------------------------------------------------------------------------: |
|         [Stanford](https://web.stanford.edu/~jurafsky/slp3/slides/LM_4.pdf)        | 
|          [Machine Learning Mastery](https://machinelearningmastery.com/)           | 
|          [Towardsdatascience](https://towardsdatascience.com/)                     | 
|          [Ofir Press](https://ofir.io/Neural-Language-Modeling-From-Scratch/)      | 
|          [Wikipedia](https://en.wikipedia.org/wiki/Language_model)                 | 
|          [scaler](https://www.scaler.com/topics/nlp/language-models-in-nlp/)       | 
|          [Attention Is All You Need](https://arxiv.org/abs/1706.03762)             | 
|          [GPTQ](https://arxiv.org/abs/2210.17323)                                  | 
|          [GGML](https://github.com/ggerganov/ggml)                                 | 




