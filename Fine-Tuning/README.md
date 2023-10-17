# Content
* **[LoRA](#lora-low-rank-adaption-for-llms)**
* **[QLoRA](#qlora-quantized-llms-and-low-rank-adaption)**
* **[QA LoRA](#qa-lora-quantized-aware-low-rank-adaption)**
* **LongLoRA**

## LoRA (Low Rank Adaption for LLMs)
> LoRA is a training method that accelerates the training of large language models while consuming less memory. It adds pairs of trainable rank-decomposition weight matrices (Called Update matrices) to existing weights, and only trains those newly added added weights.

<img align="center" src="https://images.ctfassets.net/xjan103pcp94/6fct47v2q8PU36X9A1TUzN/62bf8834293c1ec4a7e591f42ed1ffd1/pretrainined-weights-diagram-lora-blog.png">

> Method: The technique constrains the rank of the update matrix ΔW using its rank decomposition. It represents ΔWₙₖ as the product of 2 low-rank matrices Bₙᵣ and Aᵣₖ where r << min(n, k). This implies that the forward pass of the layer, originally Wx, is modified to Wx + BAx.
>
> A random Gaussian initialization is used for A and B is initially to 0, so BA=0 at the start of training. The update BA is additionally scaled with a factor α/r.

* Advantages:
  * Previos pretrained weights are kept frozen so the model is not as prone to catastrophic forgetting.
  * Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*F7uWJePoMc6Qc1O2WxmQqQ.png">

> e.g. for the previos image: let's say that d=100 , k=100 and r=5 then the original matrix size is 100 * 100 = 10000 which means 10000 parameters.
>
> But after using Rank-decomposition matrices you have 100 * 5 = 500 and 5 * 100 = 500 which means 500 + 500 = 1000 parameters and that is a huge improvement which result in less computations.

* rank-decomposition weight matrices are generally added to the attention layers of the original model.
* The greater memory-efficiency allows you to run fine-tuning on consumer GPUs like the Tesla T4, RTX 4080 or even the RTX 3080 Ti! GPUs like the T4 are free and readily accessible in Kaggle or Google Colab notebooks.


### LoRA Implementation
**1. Load the model and tokenizer**
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype = torch.float16,
    device_map = 'auto'
)

tokenizer = AutoTokenizer.from_pretrained("model_name")
```

**2. LoRA configuration**
```
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, # rank of update matrices
    lora_alpha = 16, # scaling factor
    target_modules=['query_key_value'], # The modules to apply the LoRA update matrices.
    lora_dropout = 0.05, # for regularization
    bias = 'none', # whether to train bias params or not
    task_type = 'CAUSAL_LM' # task of the model
)

model = get_peft_model(model, config)     
```
* For more about parameters selection check this [blog](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

**3. Load dataset and create prompt template**
```
from datasets import load_dataset

dataset = load_dataset('sateset_name')

# This example is for Q&A task
def create_prompt(context, question, answer) :
  if len(answer['text']) < 1 :
    answer = 'Can not find answer'
  else :
    answer = answer['text'][0]

  prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}"
  return prompt_template

mapped_qa_dataset = qa_dataset.map(lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))
```

**4. Training arguments and training**
```
import transformers
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = 'outputs',
    learning_rate = 1e-5,
    num_train_epochs = 1,
    weight_decay = 0.01,
    logging_steps=1,
    max_steps = 100,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = mapped_qa_dataset['train'],
    eval_dataset = mapped_qa_dataset['validation'],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cashe = False

trainer.train()
```

**5. inference and push to hub (optional)**
```
HUGGING_FACE_USER_NAME = 'your hugging face username'
from huggingface_hub import notebook_login
notebook_login() # enter token
model_name = "mdoel_name"

model.push_to_hub(f"{HUGGING_FACE_USER_NAME}/{model_name}", use_auth_token=True)
```
```
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
qa_model = PeftModel.from_pretrained(model, peft_model_id)
```
> [!NOTE]
> You may encounter latency issues during inference due to separately loading the base model and the LoRA model.
> To eliminate latency, use the merge_and_unload() function to merge the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.
>
> This works because during training, the smaller weight matrices (A and B in the diagram above) are separate. But once training is complete, the weights can actually be merged into a new weight matrix that is identical.
>
>> Use merge_adapter() to merge the LoRa layers into the base model while retaining the PeftModel. This will help in later unmerging, deleting, loading different adapters and so on.
>> 
>> Use unmerge_adapter() to unmerge the LoRa layers from the base model while retaining the PeftModel. This will help in later merging, deleting, loading different adapters and so on.
>>
>> Use unload() to get back the base model without the merging of the active lora modules. This will help when you want to get back the pretrained base model in some applications when you want to reset the model to its original state. For example, in Stable Diffusion WebUi, when the user wants to infer with base model post trying out LoRAs.
>>
>> Use delete_adapter() to delete an existing adapter.
>>
>> Use add_weighted_adapter() to combine multiple LoRAs into a new adapter based on the user provided weighing scheme.
>
> For more check [huggingface](https://huggingface.co/docs/peft/conceptual_guides/lora)



---
## QLoRA (Quantized LLMs and Low Rank Adaption)
<img src="https://miro.medium.com/v2/resize:fit:1400/0*oV_KwvWnFYzuWzlz.png">

> **QLoRA**, is a technique that helps in training and fine-tuning large language models (LLMs) on regular computers with limited memory. It addresses the challenge of memory requirements when working with these models.
>
> The key idea behind QLoRA is to make LLMs more efficient by reducing their memory usage while maintaining reliable performance. It achieves this through several steps: by introducing 4-bit quantization, a new data type called 4-bit NormalFloat (NF4), double quantization, and paged optimizers.

* **4-bit quantization:**
  - 4-bit quantization of weights and apply PEFT, inject LoRA adapters in each layer in 32-bit precision, and start to fine-tune the complete Language model on a specific task, **for the quantized configuration to reduce the quantization error of the system.**
  - Perform additionally a mixed precision training to balance the trade-off between accuracy and speed/memory usage.
  - QLoRA has one storage data type (NF4) and a computation data type (16-bit BrainFloat).
  - We dequantize the storage data type to the computation data type to perform the forward and backward pass, **but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat.**
* **Double Quantization:**
  - This is a technique that combines 4-bit quantization with 8-bit quantization to further reduce the memory footprint.
  - By applying a second quantization to the quantization constants, the memory footprint of these constants can be significantly reduced. 
* **Paged Optimizers:**
  - This allows QLoRa to use more memory than is available on a single GPU by paging in and out data as needed.

### QLoRA Implementation
**1. Load the model and apply 4bit quantization**
```
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments

quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", # as explained we use 4bit for the pretrained weights while using BF-16 for computations.
        bnb_4bit_compute_dtype = torch.float16,
)

model_name = "Enter your model name"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_safetensors = True,
    quantization_config = quantization_config,
    trust_remote_code = True,
    device_map = 'auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**2. LoRA configuration**
```
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha = 64,
    lora_dropout = 0.1,
    target_modules = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj",],
    bias = "none",
    task_type = "CAUSAL_LM",
)
```

**3. Load dataset and create prompt**
```
from datasets import load_dataset, Dataset

dataset = load_dataset()
```

**4. Training aruguments and trainer**
```
import transformers
from transformers import TrainingArguments
from trl import SFTTrainer


training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
```

**5. Load model, inference and apply merge_and_unload() as discussed above**
```
from peft import AutoPeftModelForCausalLM

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    low_cpu_mem_usage=True,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")
```

### QLoRA summary
* The original pre-trained weights are quantized to 4-bit and kept frozen. Then a small number of trainable parameters in the form of low-rank adpters are introduced during fine-tuning. These adapters are trained to adapt the pre-trained model to specific task in fp32 or bf16.
* When it comes to computations the 4-bit quantized weights are dequantized back to fp32.
* After fine-tuning the model consists of the original weights in 4-bit and the additional low-rank adapters in their higher precision format.
* The adpaters are in higher format for a few reasons:
  - Higher precision allows the model to capture more subtle patterns in the data. This is important for the low-rank adapters, as they are responsible for adapting the pre-trained model to the specific task it is being fine-tuned for.
  - Higher precision formats ensures that updates are accurately captured.
  - Computations with 32-bit can be faster than with lower precision.
* QLoRA backpropagates gradients through a frozen, 4-bit quantized pretraining language model into low rank adpaters (LoRA).  


---
## QA LoRA (Quantized Aware Low Rank Adaption)


    
### The following image shows the difference between the 3-methods 
<img src="https://github.com/yuhuixu1993/qa-lora/raw/main/image/qalora.png">




