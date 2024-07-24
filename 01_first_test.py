from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class MyOutputParser(StrOutputParser):
    def parse(self,text):
        return text.replace('Assistant: ','')

output_parser=MyOutputParser()

llm= Ollama(model='llama2')

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','You are a content manager with extensive SEO knowledge. Your task is to write an article based on a given title.'),
        ('user','{input}')
    ]
)
chain= prompt | llm | output_parser

print(chain.invoke({'input':'How does software change the world?'}))