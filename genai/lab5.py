from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jtest",
    enhanced_schema=True,
)

chain = GraphCypherQAChain.from_llm(
    cypher_llm=Ollama(model="llama3",temperature=0),
    qa_llm=Ollama(model="llama3",temperature=0),
    graph=graph, verbose=True,
)

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    response = chain.invoke({"query": query})
    print(response["result"])