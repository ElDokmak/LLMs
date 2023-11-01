# Quantization
---
## Content
- **[Types of Quantization](#types-of-quantization)**
- **[GPTQ: Post-training quantization on generative models](#gptq-post-training-quantization-on-generative-models)**
- **[AutoGPTQ Integration](#autogptq-integration)**
- **[GGML](#ggml)**
- **[NF4 vs. GGML vs. GPTQ](#nf4-vs-ggml-vs-gptq)**
- **[Quantization vs. Pruning vs. Knowledge Distillation](#quantization-vs-pruning-vs-knowledge-distillation)**
- **[Summary](#Summary)**



---
## Types of Quantization
1. **Post-Training Quantization (PTQ)**
   Is a straightforward technique where the weights of an already trained model are converted to lower precision without necessitating any retraining.
   Although easy to implement, PTQ is associated with potential performance degradation.
2. **Quantization-Aware Training (QAT)**
   incorporates the weight conversion process during the pre-training or fine-tuning stage, resulting in enhanced model performance. However, QAT is computationally expensive and demands representative training data.

* **We will focus on PTQ only.**



---
## GPTQ: Post-training quantization on generative models
* GPTQ is not only efficient enough to be applied to models boasting hundreds of billions of parameters, but it can also achieve remarkable precision by compressing these models to a mere 2, 3, or 4 bits per parameter without sacrificing significant accuracy.
* What sets GPTQ apart is its adoption of a mixed int4/fp16 quantization scheme. Here, model weights are quantized as int4, while activations are retained in float16. During inference, weights are dynamically dequantized, and actual computations are performed in float16.
* GPTQ has the ability to quantize models without the need to load the entire model into memory. Instead, it quantizes the model module by module, significantly reducing memory requirements during the quantization process.
* GPTQ first applies scalar quantization to the weights, followed by vector quantization of the residuals.

### **When you should use GPTQ?**
   * An approach that is being applied to numerous models and that is indicated by HuggingFace, is the following:
      - Fine-tune the original LLM model with bitsandbytes in 4-bit, nf4, and QLoRa for efficient fine-tuning.
      - Merge the adapter into the original model
      - Quantize the resulting model with GPTQ 4-bit


---
## AutoGPTQ Integration
> The AutoGPTQ library emerges as a powerful tool for quantizing Transformer models, employing the efficient GPTQ method.

* AutoGPTQ advantages : 
  - Quantized models are serializable and can be shared on the Hub.
  - GPTQ drastically reduces the memory requirements to run LLMs, while the inference latency is on par with FP16 inference. 
  - AutoGPTQ supports Exllama kernels for a wide range of architectures.
  - The integration comes with native RoCm support for AMD GPUs.
  - Finetuning with PEFT is available.



## AutoGPTQ Requirements
* Installing dependencies
```
!pip install -q -U transformers peft accelerate optimum
!pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
```

* In order to quantize your model, you need to provide a few arguemnts:
  - **bits:** The number of bits u need to quantize to.
  - **dataset:** dataset used to calibrate the quantization
  - **model_seqlen:** the model sequence length used to process the dataset
  - **block_name_to_quantize**
```
from optimum.gptq import GPTQQuantizer
quantizer = GPTQQuantizer(
      bits=4,
      dataset="c4",
      block_name_to_quantize = "model.decoder.layers",
      model_seqlen = 2048
)
```

* Save the model
```
save_folder = "/path/to/save_folder/"
quantizer.save(model,save_folder)
```

* Load quantized weights
```
from accelerate import init_empty_weights
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
empty_model.tie_weights()
quantized_model = load_quantized_model(empty_model, save_folder=save_folder, device_map="auto")
```

* Exllama kernels for faster inference
> For 4-bit model, you can use the exllama kernels in order to a faster inference speed. It is activated by default. If you want to change its value, you just need to pass disable_exllama in load_quantized_model(). In order to use these kernels, you need to have the entire model on gpus.
```
quantized_model = load_quantized_model(
      empty_model,
      save_folder=save_folder,
      device_map="auto",
      disable_exllama=False
)
```



---
## GGML
GGML is a C library focused on machine learning. It was designed to be used in conjunction with the llama.cpp library.
* The library is written in C/C++ for efficient inference of Llama models. It can load GGML models and run them on a CPU. Originally, this was the main difference with GPTQ models, which are loaded and run on a GPU.

### Quantization with GGML
The way GGML quantizes weights is not as sophisticated as GPTQ’s. Basically, it groups blocks of values and rounds them to a lower precision. Some techniques, like Q4_K_M and Q5_K_M, implement a higher precision for critical layers. In this case, every weight is stored in 4-bit precision, with the exception of half of the attention.wv and feed_forward.w2 tensors.

* Experimentally, this mixed precision proves to be a good tradeoff between accuracy and resource usage.
* weights are processed in blocks, each consisting of 32 values. For each block, a scale factor (delta) is derived from the largest weight value. All weights in the block are then scaled, quantized, and packed efficiently for storage (nibbles).
* This approach significantly reduces the storage requirements while allowing for a relatively simple and deterministic conversion between the original and quantized weights.



---
## NF4 vs. GGML vs. GPTQ
<kbd>
   <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*yz7rSvjKtukVXdVHwxGfAQ.png">
</kbd>

* Based on these results, we can say that GGML models have a slight advantage in terms of perplexity. The difference is not particularly significant, which is why it is better to focus on the generation speed in terms of tokens/second.
* The best technique depends on your GPU: if you have enough VRAM to fit the entire quantized model, GPTQ with ExLlama will be the fastest. If that’s not the case, you can offload some layers and use GGML models with llama.cpp to run your LLM.



--- 
## Quantization vs. Pruning vs. Knowledge Distillation
## Pruning
> Removing some of the connections in our neural network, which result in **sparse network**.
<kbd>
   <img src="https://miro.medium.com/v2/resize:fit:559/1*0mGJU7eNWgrqU5sgk-7RoQ.png">
</kbd>

* Procedure:
  * Pick pruning factor X.
  * In each layer, set the lowest X% of weights to zero.
  * Optional: retrain the model to recover accuracy.

### General MatMul vs. Sparse MatMul
<kbd>
   <img src="https://github.com/ElDokmak/LLMs/assets/85394315/c7a3526e-995b-43e7-8a99-6b1e43883826">
</kbd>

* General MatMul: we multiply every row and colum by each other even they were zeros.
* Sparse MatMul: we skip rows and columns with zeros, we multiply only ones with values in it.

## Distillation
> We train a **Student model** on the output labels of the **Teacher model**
<kbd>
   <img src="https://github.com/ElDokmak/LLMs/assets/85394315/160351cd-1a97-450d-b8f8-ab9a9dac65f0">
</kbd>



---
## Summary
<img src="https://github.com/ElDokmak/LLMs/assets/85394315/15cf3903-455c-4811-8f7e-14b6eaf30237">

---
