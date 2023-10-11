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

> In both cases the goal is to map FP32 into INT8.

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

> e.g. We have a maximum value of 3.2 and a minimum value of -3.0. We can calculate the scale is 255/(3.2 + 3.0) = 41.13 and the zero-point -round(41.13 × -3.0) - 128 = 123 -128 = -5, so our previous weight of 0.1 would be quantized to round(41.13 × 0.1 -5) = -1.
> This is very different from the previous value obtained using absmax (4 vs. -1).
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*n5nqoJUXp65JahKsLzQS-A.png">
-->


###  GPTQ: Post-training quantization on generative models
> GPTQ is not only efficient enough to be applied to models boasting hundreds of billions of parameters, but it can also achieve remarkable precision by compressing these models to a mere 2, 3, or 4 bits per parameter without sacrificing significant accuracy.

> What sets GPTQ apart is its adoption of a mixed int4/fp16 quantization scheme. Here, model weights are quantized as int4, while activations are retained in float16. During inference, weights are dynamically dequantized, and actual computations are performed in float16.

> GPTQ has the ability to quantize models without the need to load the entire model into memory. Instead, it quantizes the model module by module, significantly reducing memory requirements during the quantization process.

> GPTQ first applies scalar quantization to the weights, followed by vector quantization of the residuals. 

#### **When you should use GPTQ?**
   - An approach that is being applied to numerous models and that is indicated by HuggingFace, is the following:
      - Fine-tune the original LLM model with bitsandbytes in 4-bit, nf4, and QLoRa for efficient fine-tuning.
      - Merge the adapter into the original model
      - Quantize the resulting model with GPTQ 4-bit


### AutoGPTQ 
> The AutoGPTQ library emerges as a powerful tool for quantizing Transformer models, employing the efficient GPTQ method.

>  AutoGPTQ advantages :
> Quantized models are serializable and can be shared on the Hub.
> 
> GPTQ drastically reduces the memory requirements to run LLMs, while the inference latency is on par with FP16 inference.
> 
> AutoGPTQ supports Exllama kernels for a wide range of architectures.
> 
> The integration comes with native RoCm support for AMD GPUs.
> 
> Finetuning with PEFT is available.


### GGML
> GGML is a C library focused on machine learning. It was designed to be used in conjunction with the llama.cpp library.
>
> The library is written in C/C++ for efficient inference of Llama models. It can load GGML models and run them on a CPU. Originally, this was the main difference with GPTQ models, which are loaded and run on a GPU.

#### **Quantization with GGML**
> The way GGML quantizes weights is not as sophisticated as GPTQ’s. Basically, it groups blocks of values and rounds them to a lower precision. Some techniques, like Q4_K_M and Q5_K_M, implement a higher precision for critical layers. In this case, every weight is stored in 4-bit precision, with the exception of half of the attention.wv and feed_forward.w2 tensors.

> Experimentally, this mixed precision proves to be a good tradeoff between accuracy and resource usage.
>
>  weights are processed in blocks, each consisting of 32 values. For each block, a scale factor (delta) is derived from the largest weight value. All weights in the block are then scaled, quantized, and packed efficiently for storage (nibbles).
>
> This approach significantly reduces the storage requirements while allowing for a relatively simple and deterministic conversion between the original and quantized weights.


#### **NF4 vs. GGML vs. GPTQ**
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*yz7rSvjKtukVXdVHwxGfAQ.png">

> Based on these results, we can say that GGML models have a slight advantage in terms of perplexity. The difference is not particularly significant, which is why it is better to focus on the generation speed in terms of tokens/second.

> The best technique depends on your GPU: if you have enough VRAM to fit the entire quantized model, GPTQ with ExLlama will be the fastest. If that’s not the case, you can offload some layers and use GGML models with llama.cpp to run your LLM.



















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




