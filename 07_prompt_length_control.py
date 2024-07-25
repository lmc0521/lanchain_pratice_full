from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_history=[]
llm = Ollama(model='llama2')
# llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt=ChatPromptTemplate.from_messages(
    [
        ('system','You are a powerful chat bot.'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user','{input}')
    ]
)

def condense_prompt(prompt:ChatPromptValue) -> ChatPromptValue:
    messages=prompt.to_messages()
    num_tokens=llm.get_num_tokens_from_messages(messages)
    recent_messages=messages[2:]
    while num_tokens > 500:
        recent_messages=recent_messages[2:]
        num_tokens=llm.get_num_tokens_from_messages(
            messages[:2]+recent_messages
        )
    chat_history=recent_messages
    # print(chat_history)
    messages=messages[:2]+recent_messages
    return ChatPromptValue(messages=messages)

chain=prompt | condense_prompt | llm

input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = chain.invoke({
            'input': input_text,
            'chat_history': chat_history,
        })
        chat_history.append(HumanMessage(content=input_text))
        chat_history.append(AIMessage(content=response))
        print(response)
    input_text = input('>>> ')