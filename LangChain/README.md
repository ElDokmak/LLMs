# Content
* **[Language models/ Prompts/ Output parsers](#models-prompts-output-parsers)**




---
## ***Models/ Prompts/ Output parsers***
> The core element of any language model application

## Prompts
A prompt refers to the statement or question provided to the LLM to request information.
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*6q8_6ZjOWb58z3hhprzwpA.png">

* **Text Prompt Templates**
```
from langchain.prompts import PromptTemplate

template = """You will provided with the sample text. \
Your task is to rewrite the text to be gramatically correct. \
Sample text: ```{sample_text}``` \
Output: 
"""
prompt_template = PromptTemplate.from_template(template = template)

sample_text = "Me likes cats not dogs. They jumps high so much!"

final_prompt = prompt_template.format(sample_text = sample_text)
print(final_prompt)
```

* **Chat prompt templates**
```
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

system_template = """You will provided with the sample text. \
Your task is to translate the text into {output_language} language \
and summarize the translated text in at most {max_words} words. \ 
"""

system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{sample_text}"
human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt_template, human_message_prompt_template])

final_prompt = chat_prompt_template.format_prompt(
  output_language="English",
  max_words=15,
  sample_text="Estoy deseando que llegue el fin de semana."
).to_messages()
print(final_prompt)
```

## Models
<img src="https://miro.medium.com/v2/resize:fit:786/format:webp/1*2ZVRlCJMvg6HmM_OwL8P_Q.png">

### Language models

- **LLMs:** inputs and outputs text
```
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name= "gpt-3.5-turbo" , temperature= 0)
tempalte = """You will provided with the sample text. \
Your task is to rewrite the text to be gramatically correct. \
Sample text: ```{sample_text}``` \
Output: 
"""

prompt_template = PromptTemplate.from_template(template=template)

sample_text = "Me likes cats not dogs. They jumps high so much!"
final_prompt = prompt_template.format(sample_text=sample_text)

completion = llm(final_prompt)
print(completion)
```

* **Chat models:** inputs and outputs chat messages
```
from langchain.chat_models impots ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(temperature=0)
system_template = """You will provided with the sample text. \
Your task is to translate the text into {output_language} language \
and summarize the translated text in at most {max_words} words. \ 
"""

system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{sample_text}"
human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt_template, human_message_prompt_template])
final_prompt = chat_prompt_template.format_prompt(
  output_language="English",
  max_words=15,
  sample_text="Estoy deseando que llegue el fin de semana."
).to_messages()

completion = chat(final_prompt)
print(completion)
```

### Text Embedding Models
Text embedding is used to represent text data in a numerical format that can be understood and processed by ML models.

```
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")
text = "It is imperative that we work towards sustainable practices, reducing waste and conserving resources."

embedded_text = embeddings.embed_query(text)
print(embedded_text)
```

## Parsers
Outpus Parsers help get structure responses.

---
## 







