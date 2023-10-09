# Transformers
<!--
<img src="https://miro.medium.com/v2/resize:fit:1400/1*10K7SmGoJ5zAtjkGfNfjkg.png">
-->
## Content
* **[Input Embeddings](#Embeddings)**
* **[Postitional Encoding](#Postional-Encoding)**
* **[Self Attention](#Self-Attention)**
* **[Multi-Head Attention](#Multi-Head-Attention)**
* **[Layer Normalization/ Residual Connections/ Feed Forward Network](#layer-normalization-residual-connections-feed-forward-network)**
* **Encoder**
* **Decoder**


---
## Embeddings
> Word Embedding can be thought of as a learned vector representation of each word. A vector which captures contextual information of these words.

> Neural networks learn through numbers so each word maps to a vector with continuous values to represent that word.
<img align="center" width=300 src="https://miro.medium.com/v2/resize:fit:582/format:webp/0*6MnniQMOBPu4kFq3.png">


---
## Postional Encoding
> Embedding represents token in a d-dimensional space where tokens with similar meaning are closer to each other. However, these embeddings don't encode the relative position of the tokens in a sentence.

> Same as the name Postional Encodding encodes the postion of the words in the sequence.

* Formula for calculating the positional encoding is :
<img width="500" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*z3Rfl0wGsBsaZcprpqR8Uw.png">

Postional Encoding works because absolute position is less important than relative position.
> e.g. "My fried chicken was very good" 
we do not know that the word "good" is at index 6 and the word “looks” is at index 5. It’s sufficient to remember that the word “good” tends to follows the word “looks”.


---
## Self Attention (Scaled Dot-Product Attention)
<img src="https://miro.medium.com/v2/resize:fit:640/0*NEPbOP47PlMTXoIb">
After feeding the query, key, and value vectors through a linear layer, we calculate the dot product of the query and key vectors. 
The values in the resulting matrix determine how much attention should be payed to the other words in the sequence given the current word.

In other words, each word (row) will have an attention score for every other word (column) in the sequence.
> e.g. "On the river bank"
<img src="https://miro.medium.com/v2/resize:fit:640/0*8BFdH5nY0KoLq1I8">

> For short, you can think of attention as which part of the sentence should I focus on.

> The dot-product is scaled by a square root of the depth.
This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients which make it difficult to learn.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*WmNb5ugFkawpvSqYqvTaNg.png">

> Then apply softmax to obtain values between 0 and 1.

<img src="https://miro.medium.com/v2/resize:fit:640/0*KgHYTtBIJcn8sNUZ">

> Then multiply by the value vector.

<img src="https://miro.medium.com/v2/resize:fit:828/0*G-mOxnggdLyNlc8y">


---
## Multi-Head Attention
Instead of one single head attention, Q, K, and V are split into multiple heads. Which allows the model to jointly attend to information from different representation subspaces at different positions.

> e.g. "On the river bank" for the first head the word "The" will attend the word "bank" while for the second head the word "The" will attend to the word "river"
<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*Nc9uK_gEwm18OrNmwP_lBg.png">

> Note: After splitting the total computation ramains the same as a single head.

> The attention output is concatenated and put through a Dense layer.
<img src="https://miro.medium.com/v2/resize:fit:640/0*j_1WMIQxKPh-L7OY">



---
## Layer Normalization/ Residual Connections/ Feed Forward Network
> Residual Connections = Input Embedding + Positional Encodding are added to Multi-Head Attention.

> Normalization means having Mean = 0 and variance = 1.

* Residual Connections help avoiding the vanishing gradient problem in deep nets.
* Each hidden layer has a residual connection around it followed by a layer normalization.
<img src="https://miro.medium.com/v2/resize:fit:640/0*21NPCniNISCCVfxn">

* Then the ouput finishes by passing through a point wise feed forward network.
<img src="https://miro.medium.com/v2/resize:fit:622/format:webp/1*ItvJ0KeOKCFSXDUNce2zAA.png">



---
## Encoder
> The Encoder's job is to map all input sequences into an abstract continous representation that holds information.
<img src="https://miro.medium.com/v2/resize:fit:640/0*K67VOXrh_xgyCiHS">



---
## Decoder
