# Content
* **[Language models/ Prompts/ Output parsers](#models-prompts-output-parsers)**
* **[Chains](#chains)**
* **[Memory](#memory)**




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
```
gift_schema = ResponseSchema(
    name = "gift",
    description="Was the item purchased\
    as a gift or somene else?\
    Answer True if yes,\
    False if not or unknown."
)

delivery_days_schema = ResponseSchema(
    name = "delivery_days",
    description = "How many days\
    did it take for the product\
    to arrive? If this \
    information is not found,\
    output -1."
)

price_value_schema = ResponseSchema(
    name="price_value",
    description="Extract any\
    sentences about the value or \
    price, and output them as a \
    comma separated Python list."
)

response_schemas = [
    gift_schema,
    delivery_days_schema,
    price_value_schema
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
```



---
## ***Chains***
While using a single LLM may be sufficient for simpler tasks, LangChain provides a standard interface and some commonly used implementations for chaining LLMs together for more complex applications, either among themselves or with other specialized modules. Or you can think simple chain can be defined as sequence of calls.

### **Types of chains:**
- **LLMChain:** Simple chain that consists of PromptTemplate and model
```
from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is capital of {country}?"
llm = OpenAI(temperature = 0)
llm_chain = LLMChain(
  llm = llm,
  prompt = PromptTemplate.from_template(prompt_template)
)

llm_chain("Egypt")
```
- **Sequential Chains:** Combine multiple chains where the output of one chain is the input of the next one.
  - **There are 2 types:**
    1. __SimpleSequentialChain:__ Single input/ output
    2. __SequentialChain:__ Multiple inputs/ outputs

```
from langchain.chains import SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.9)


# Prompt1
first_prompt = ChatPromptTemplate.from_template(
    "What is the capital of {country}?"
)
chain_one = LLMChain(llm= llm, prompt = first_prompt)

# prompt2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    country: {capital}"
)
chain_two = LLMChain(llm= llm, prompt=second_prompt)

# Simple sequential chain
simple_chain = SimpleSequentialChain(
    chains = [chain_one, chain_two],
    verbose = True   # This line shows the process
)

simple_chain.run("Egypt")
```

<img src="https://miro.medium.com/v2/resize:fit:1400/1*jJE0uZTBadEYqe0hEXlWuQ.png">

- **Router Chain**
<img width="800" src="https://miro.medium.com/v2/resize:fit:1400/1*d_7dKnR9W2NwSwq5S5DN1g.png">



---
## Memory
> Large Language Models are **stateless** each transaction is independent.
> 
> Chatbots have memory by providing conversation as context. Memory allowas LLM to remember previos interactions with the user

<img width="800" src="https://cdn.sanity.io/images/vr8gru94/production/927ca8cc5d92ee75f36d7eb4bef4685c4e3118e5-2880x1370.png">

* **Memory types**
  - **ConversationBufferMemory:** Allows for storing of messages and then extract the messages in a variable.
    ```
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory()
    memory.save_context({"input": "hi"}, {"output": "whats up"})

    memory.load_memory_variables({})
    ```

  - **ConversationBufferWindowMemory:** Keeps a list of the interactions of the conversation over time, it only uses the last K interactions.
    ```
    from langchain.memory import ConversationBufferWindowMemory

    memory = ConversationBufferWindowMemory(k = 1) # saves the last message only
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})

    memory.load_memory_variables({}) # it will load the last one as k = 1
    ```
    
  - **ConversationTokenBufferMemory:** Keeps a buffer of recent interactions in memory, and uses token length rather than number of interactions to determine when to flush interactions.
    ```
    from langchain.memory import ConversationTokenBufferMemory
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI
    llm = ChatOpenAI(temperature=0.0)

    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
    memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
    memory.save_context({"input": "Chatbots are what?"}, 
                        {"output": "Charming!"})
    
    memory.load_memory_variables({})
    ```
    
  - **ConversationSummaryMemory:** Creates a summary of the conversation over time.
    ```
    from langchain.llms import OpenAI
    from langchain.chains import ConversationChain
    
    llm = OpenAI(temperature=0)
    conversation_with_summary = ConversationChain(
      llm=llm,
      memory=ConversationSummaryMemory(llm=OpenAI()),
      verbose=True
    )
    
    conversation_with_summary.predict(input="Hi, what's up?")
    ```
    
  - **ConversationSummaryBufferMemory:** Saves the latest X interactions as well as their summary.
    ```
    from langchain.memory import ConversationSummaryBufferMemory

    schedule = "Write whatever u want"
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})

    conversation = ConversationChain(
        llm=llm, 
        memory = memory,
        verbose=True
    )
    
    conversation.predict(input="Ask about something in the schedule")
    ```
  - **ConversationEntityMemory:** Saves information about entities.
    ```
    from langchain.memory import ConversationEntityMemory

    llm = OpenAI(temperature = 0)
    memory = ConversationEntityMemory(llm = llm)
    _input = {"input": "Ahmed and Anas are working on Google Brain"}
    memory.load_memory_variables(_input)
    memory.save_context(
        _input,
        {"output": " That sounds like a great postion! What kind of work ther are doing there?"}
    )

    memory.load_memory_variables({"input": 'who is Ahmed'})
    ```
    
  - **VectorStoreRetreiverMemory:** Stores interactions in VectorDB and finds the top-K most similar documents every time it is called.

We can use multiple memories at the same time.














