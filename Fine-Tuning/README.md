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

### Implementation
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

## QLoRA (Quantized Low Rank Adaption)




