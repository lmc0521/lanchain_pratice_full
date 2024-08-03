from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

question='Explain how AlphaCodium work?s'
web_results = web_search_tool.invoke({"query": question})
print(web_results)