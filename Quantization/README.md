# Quantization
## AutoGPTQ Integration 
> With GPTQ you can quantize your language model to 8, 4, 3,or even 2 bits. Without big drop in performance and with faster inference speed.

## Requirements
* Installing dependencies
```
!pip install -q -U transformers peft accelerate optimum
!pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
```

* In order to quantize your model, you need to provide a few arguemnts:
  * **bits:** The number of bits u need to quantize to.
  * **dataset:** dataset used to calibrate the quantization
  * **model_seqlen:** the model sequence length used to process the dataset
  * **block_name_to_quantize**
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
