from py2neo import Graph
 
# Connect to the local Neo4j instance
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "neo4jtest"))
 
def query_graph():
    query = """
    MATCH (p:Person)-[:WORKED_WITH]->(p2:Person)
    RETURN p.name AS person, p.profession AS profession, p2.name AS collaborator
    """
    result = graph.run(query).data()
    return result
 
result = query_graph()

import requests

# Use Ollama to generate a detailed response based on the retrieved answer
ollama_url = "http://localhost:11434/api/generate"

# Prepare a prompt from the graph result
prompt = "Generate a biography based on the following information:\n\n"
for record in result:
    prompt += f"{record['person']} is a {record['profession']} who worked with {record['collaborator']}.\n"
 
# Use Ollama for generation
query = {
    "model": "llama3",
    "prompt": prompt,
    "stream": False
}
 
response = requests.post(ollama_url, json=query)
 
# Check if the request was successful
if response.status_code == 200:
    # Process the response
    if query["stream"]:
        # Handle streaming response
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                print(body.get("response", ""))
    else:
        # Handle non-streaming response
        response_data = response.json()
        print(response_data["response"])
else:
    # Print error message
    print("Error:", response.status_code, response.text)
