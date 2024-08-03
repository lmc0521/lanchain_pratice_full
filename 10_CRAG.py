import pprint
import uuid
from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma, FAISS, SKLearnVectorStore
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
local_llm="llama3"

#Load
url="https://lilianweng.github.io/posts/2023-06-23-agent/"
loader=WebBaseLoader(url)
docs=loader.load()

# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)


embedding = OpenAIEmbeddings()

# vectordb=FAISS.from_documents(docs,embedding)
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=embedding,
)

retriever=vectorstore.as_retriever()

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    You will be given: \n
    1/ a question
    2/ a retrieved document    

    question: {question} \n
    retrieved document: \n\n {documents} \n\n
    
    If the document contains keywords related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
    input_variables=["question", "documents"],
)
'''
prompt= PromptTemplate(
    templete="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {documents} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
    input_variables=["question", "documents"],
)
'''
retrieval_grader = prompt | llm | JsonOutputParser()
# question = "Explan how the different type of agent memory work?"
# question="Explain how AlphaCodium works?"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# print(retrieval_grader.invoke({"question": question, "documents": doc_txt}))

# Prompt
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["question", "documents"],
)

# LLM
llm = ChatOllama(model=local_llm, temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
# generation = rag_chain.invoke({"documents": docs, "question": question})
# print(generation)

### Search

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print('---RETRIEVE---')
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print('---GENERATE---')
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print('---CHECK RELEVENCE---')
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    points = 0
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score["score"]
        print(grade)

        if grade == "yes":
            print('---GRADDE: DOCUMENT RELEVANT---')
            filtered_docs.append(d)
            points+=1
        else:
            print('---GRADDE: DOCUMENT NOT RELEVANT---')
            # search = "Yes"
            # continue
            points-=1
    print(points)
    if points<0:
        search="Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print('---WEB SEARCH---')
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    web_results = web_search_tool.invoke({"query": question})
    documents.extend(
        [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]
    )
    return {"documents": documents, "question": question, "steps": steps}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print('---DECIDE TO GENERATE---')
    search = state["search"]
    if search == "Yes":
        print('---DECISION: RUN WEB SEARCH---')
        return "search"
    else:
        print('---DECISION: GENERATE---')
        return "generate"


# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

custom_graph = workflow.compile()

display(Image(custom_graph.get_graph(xray=True).draw_mermaid_png()))

#Run

inputs={
        "question":"Explan how the different type of agent memory work?",
        "steps": []
}
# inputs={
#         "question":"what is agent memory?",
#         "steps": []
# }
# inputs={
#         "question":"Explain how AlphaCodium works?",
#         "steps": []
# }

for output in custom_graph.stream(inputs):
    for key,value in output.items():
        pprint.pprint(f"Node '{key}':")
    pprint.pprint("\n---\n")
pprint.pprint(value['generation'])


'''
def predict_custom_agent_local_answer(example: dict):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = custom_graph.invoke(
        {"question": example["input"], "steps": []}, config
    )
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}


example = {"input": "What are the types of agent memory?"}
response = predict_custom_agent_local_answer(example)
print(response)
'''