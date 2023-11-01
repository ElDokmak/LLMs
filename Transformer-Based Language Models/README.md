# Transformers
<img src="https://miro.medium.com/v2/resize:fit:1400/1*10K7SmGoJ5zAtjkGfNfjkg.png">

---
## Content
* **[Input Embeddings](#Embeddings)**
* **[Postitional Encoding](#Postional-Encoding)**
* **[Self Attention](#Self-Attention)**
* **[Multi-Head Attention](#Multi-Head-Attention)**
* **[Layer Normalization/ Residual Connections/ Feed Forward Network](#layer-normalization-residual-connections-feed-forward-network)**
* **[Encoder](#Encoder)**
* **[Decoder](#Decoder)**



---
## Embeddings
Word Embedding can be thought of as a learned vector representation of each word. A vector which captures contextual information of these words.
* Neural networks learn through numbers so each word maps to a vector with continuous values to represent that word.
<kbd>
  <img width=300 src="https://miro.medium.com/v2/resize:fit:582/format:webp/0*6MnniQMOBPu4kFq3.png">
</kbd>



---
## Postional Encoding
Embedding represents token in a d-dimensional space where tokens with similar meaning are closer to each other. However, these embeddings don't encode the relative position of the tokens in a sentence.
* Same as the name Postional Encodding encodes the postion of the words in the sequence.
* Formula for calculating the positional encoding is :
<img width="500" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*z3Rfl0wGsBsaZcprpqR8Uw.png">

* Postional Encoding works because absolute position is less important than relative position.
  > e.g. "My fried chicken was very good" 
  > We do not know that the word "good" is at index 6 and the word “looks” is at index 5. It’s sufficient to remember that the word “good” tends to follows the word “looks”.



---
## Self Attention (Scaled Dot-Product Attention)
<kbd>
  <img width="500" src="https://miro.medium.com/v2/resize:fit:640/0*NEPbOP47PlMTXoIb">
</kbd>

After feeding the query, key, and value vectors through a linear layer, we calculate the dot product of the query and key vectors. 
The values in the resulting matrix determine how much attention should be payed to the other words in the sequence given the current word.
* In other words, each word (row) will have an attention score for every other word (column) in the sequence.
> e.g. "On the river bank"
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*8BFdH5nY0KoLq1I8">
</kbd>

* For short, you can think of attention as which part of the sentence should I focus on.
* The dot-product is scaled by a square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients which make it difficult to learn.
  
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*WmNb5ugFkawpvSqYqvTaNg.png">
</kbd>

* Then apply softmax to obtain values between 0 and 1.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*KgHYTtBIJcn8sNUZ">
</kbd>

* Then multiply by the value vector.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/0*G-mOxnggdLyNlc8y">
</kbd>



---
## Multi-Head Attention
Instead of one single head attention, Q, K, and V are split into multiple heads. Which allows the model to jointly attend to information from different representation subspaces at different positions.

> e.g. "On the river bank" for the first head the word "The" will attend the word "bank" while for the second head the word "The" will attend to the word "river"
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*Nc9uK_gEwm18OrNmwP_lBg.png">
</kbd>

* **Note:** After splitting the total computation ramains the same as a single head.

* The attention output is concatenated and put through a Dense layer.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*j_1WMIQxKPh-L7OY">
</kbd>



---
## Layer Normalization/ Residual Connections/ Feed Forward Network
Residual Connections = Input Embedding + Positional Encodding are added to Multi-Head Attention.

* Normalization means having Mean = 0 and variance = 1.
* Residual Connections help avoiding the vanishing gradient problem in deep nets.
* Each hidden layer has a residual connection around it followed by a layer normalization.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*21NPCniNISCCVfxn">
</kbd>

* Then the ouput finishes by passing through a point wise feed forward network.
<kbd>
<img src="https://miro.medium.com/v2/resize:fit:622/format:webp/1*ItvJ0KeOKCFSXDUNce2zAA.png">
</kbd>



---
## Encoder
The Encoder's job is to map all input sequences into an abstract continous representation that holds information.
* You can stack the encoder N times to further encode the information, where each layer has the opportunity to learn different attention representations therefore potentially boosting the predictive power of the transformer network.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*K67VOXrh_xgyCiHS">
</kbd>



---
## Decoder
The Decoder's job is to generate text sequences.

<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:640/0*lx0m8R-k0dwq7sy3">
</kbd>

* It has two multi-headed attention layers, a pointwise feed-forward layer, and residual connections, and layer normalization after each sub-layer.
* These sub-layers behave similarly to the layers in the encoder but each multi-headed attention layer has a different job.
* The decoder is capped off with a linear layer that acts as a classifier, and a softmax to get the word probabilities.
* The deocoder is autoregressive, it begins with a start token, and takes previos output as inputs, as well as the encoder outputs that contain attention information from the input.
* Decoder's Input Embeddings & Positional Encoding is almost the same as the Encoder.


### Decoder's Multi-Head Attention
The second one operates just like the Encoder while the first one operates slight different than the Encoder since the decoder is autoregressive  and generates the seq word by word, you need to prevent it from conditioning to future tokens using **Masking**.
* Mask is a matrix that’s the same size as the attention scores filled with values of 0’s and negative infinities. When you add the mask to the scaled attention scores, you get a matrix of the scores, with the top right triangle filled with negativity infinities.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*QYFua-iIKp5jZLNT.png">
</kbd>

* The reason for the mask is because once you take the softmax of the masked scores, the negative infinities get zeroed out, leaving zero attention scores for future tokens.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*3ykVCJ9okbgB0uUR.png">
</kbd>


### Linear Classifier and Final Softmax
* The output of the final pointwise feedforward layer goes through a final linear layer, that acts as a classifier.
* The output of the classifier then gets fed into a softmax layer, which will produce probability scores between 0 and 1. We take the index of the highest probability score, and that equals our predicted word.
<kbd>
  <img src="https://miro.medium.com/v2/resize:fit:786/format:webp/0*1OyVUO-s-uBh8EV2.png">
</kbd>

* The decoder then takes the output, add’s it to the list of decoder inputs, and continues decoding again until a token is predicted. For our case, the highest probability prediction is the final class which is assigned to the end token.
* The decoder can also be stacked N layers high, each layer taking in inputs from the encoder and the layers before it. By stacking the layers, the model can learn to extract and focus on different combinations of attention from its attention heads, potentially boosting its predictive power.
