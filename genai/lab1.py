import requests
import json

# Read the text from the file
with open("../data/kb.txt", "r") as file:
    knowledge_data = file.read()

# Define the local LLM endpoint (Ollama)
ollama_url = "http://localhost:11434/api/generate"
 
# Request headers
headers = {
    "Content-Type": "application/json"
}


# Define the input for the LLM
data = {
    "model": "llama3",  # The local model you pulled
    "prompt": f"Based on the following information, explain the benefits of Python:\n\n{knowledge_data}",
    "stream": False
}
 
# Send the POST request
response = requests.post(ollama_url, headers=headers, data=json.dumps(data))

# Check if the request was successful
if response.status_code == 200:
    # Process the response
    if data["stream"]:
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
