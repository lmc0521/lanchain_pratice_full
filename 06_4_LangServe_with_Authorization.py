from fastapi import Request,Depends, FastAPI, Header, HTTPException
from typing import Optional, Dict, Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.runnables import ConfigurableField, RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from langserve import add_routes





chat001 = ChatMessageHistory()
chat001.add_user_message('My name is Amo.')

store = {
    'amo': chat001,
}

def get_chat_history(user_id: str) -> BaseChatMessageHistory:
    if user_id not in store:
        store[user_id] = ChatMessageHistory()
    return store[user_id]

llm = Ollama(model='llama2').configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    ),
)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a powerful assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
])

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User Id",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
    ],
)

def per_request_config_modifier(
    config: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Update the config"""
    config = config.copy()
    configurable = config.get("configurable", {})
    user_id = getattr(request.state, 'user_id', None)
    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail="No user id found. Please set a state named 'user_id'.",
        )
    configurable["user_id"] = user_id
    config["configurable"] = configurable
    return config
async def verify_api_key(request: Request,authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # assuming the token is provided as a Bearer token
    api_key = authorization.split(" ")[1] if len(authorization.split(" ")) == 2 else None
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    if api_key != "valid_api_key":
        raise HTTPException(status_code=403, detail="Invalid API Key")

    request.state.user_id = 'amo'


add_routes(
    app,
    with_message_history,
    per_req_config_modifier=per_request_config_modifier,
    path="/llama2",
    dependencies=[Depends(verify_api_key)],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)