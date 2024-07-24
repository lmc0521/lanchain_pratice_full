from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

# llm= Ollama(model='llama2',temperature=0).configurable_fields(
#     temperature=ConfigurableField(
#         id='llm_temperature',
#         name='LLM Temperature',
#         description='The temperature of the LLM'
#     ),
#     model=ConfigurableField(
#         id='model',
#         name='The Model',
#         description="The language model"
#     )
# )
llm = Ollama(model='llama2').configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key='llama2',
    gpt35=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
)

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ('user','{input}')
#     ]
# )
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
)

chain=prompt | llm
# print(chain.with_config(configurable={'model':'codellama','llm_temperature':0.9}).invoke({'input':'Tell me a joke'}))
# print(chain.with_config(configurable={"llm": "gpt35"}).invoke({'input': 'Tell me a joke'}))
print(chain.with_config(configurable={'llm':'llama2','prompt':'poem'}).invoke({'topic': 'Earth'}))