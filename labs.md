# Gen AI: Understanding and Using RAG
## Making LLMs smarter by pairing your data with Gen AI
## Session labs 
## Revision 2.5 - 04/21/25

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Implementing basic RAG with prompt context**

**Purpose: In this lab, we’ll see a basic example of augmenting a prompt by retrieving context from a data file.**

1. First, download and install Ollama to allow us to have a local LLM to query. Go to the *TERMINAL* tab in the bottom part of the codespace and execute the command below. Then you can run the actual application to see usage.
```
curl -fsSL https://ollama.com/install.sh | sh
ollama
```
![downloading ollama](./images/rag05.png?raw=true "downloading ollama")
![ollama usage](./images/rag05a.png?raw=true "ollama usage")

2. Next, start the ollama server running in the background with the first command below. After that is done, you can just hit Enter and run the second command to pull down the *llama3* model with the second command.
```
ollama serve &
ollama pull llama3
```
![serve and pull](./images/rag06a.png?raw=true "serve and pull")

3. While this is running, go ahead and open a second terminal session. In the codespace, right-click and select the *Split Terminal* option. This will add a second terminal side-by-side with your other one. 

![splitting terminal](./images/rag08.png?raw=true "splitting terminal")
   
4. In our repository, we have a set of Python programs to help us illustrate and work with concepts in the labs. These are mostly in the *genai* subdirectory. In a free terminal session, change into that directory. (Remaining steps will assume that directory unless otherwise specified.)
```
cd genai
```

5. For this lab, we have a simple Python program that queries the LLM about the benefits of Python in general. The file name is lab1.py. Open the file either by clicking on [**genai/lab1.py**](./genai/lab1.py) or by entering the command below in the codespace's terminal.
```
code lab1.py
```

6. You can look around this file to see how it works. It simply passes a direct prompt to the LLM and returns the results. Notice the prompt *"Explain the benefits of Python"* on line 17. At line 22, you can see the request made to Ollama (and the Llama3 model.)

![examining lab1 file](./images/rag04.png?raw=true "examining lab1 file")

7. When you are done examining the code, you can go ahead and run it with the command below. 
```
python lab1.py
```
![running lab1 file](./images/rag09.png?raw=true "running lab1 file")

8. When this is done running, you'll be able to see a rather large amount of output about the benefits of python. (While this is running, you can go ahead and proceed with steps 9-11. You can click on the "+" sign (circled in the figure) to open another terminal. (You can navigate between terminals using the *tree* list of *bash* instances on the far right. Just click on the one you want.)

![lab output 1](./images/rag10a.png?raw=true "lab output 1")
  
9. Now let's update the program to add context from a local *knowledge base*. For purposes of keeping things simple and quick, we have a local file that we'll use for the additional context. The file name is *data/kb.txt*. Open the file either by clicking on [**data/kb.txt**](./data/kb.txt) or by entering the command below in the codespace's terminal.  

```
cd genai (if not there)
code ../data/kb.txt
```

10. Now update the python program to pass the context from the data file. Switch back to the *lab1.py* file. First, after the imports, add the code below to read in the data file.

```
# Read the text from the file
with open("../data/kb.txt", "r") as file:
    knowledge_data = file.read()
```
![mod 1](./images/rag11.png?raw=true "mod 1")

11. Scroll down to where the prompt is defined (probably around line 19 or 20) and modify that line to use the prompt below. Notice that it adds on the "{knowledge_data}" section, which is the content we read in. It also sets context for the LLM "Based on the following information" which refers to the knowledge base again.

```
    "prompt": f"Based on the following information, explain the benefits of Python:\n\n{knowledge_data}",
```
![mod 2](./images/rag12.png?raw=true "mod 2")

12. Save your changes to the file with CMD+S or CTRL+S (or use the three bar menu in the top left of the codespace and drill-down to the File->Save option). Then run the program again.

```
(save file)
python lab1.py
```
13. If you haven't looked at the output from the first run, you can switch back to the shell where that is to see that output.

![lab output 1](./images/rag28.png?raw=true "lab output 1")
    
14. After the second run is done, you can look in that terminal and should see output that indicates the AI referenced the data in the file to help answer the prompt.

![run 2](./images/rag29.png?raw=true "run 2") 


<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 2 - Working with Vector Databases**

**Purpose: In this lab, we’ll learn about how to use vector databases for storing supporting data and doing similarity searches.**

1. In our repository, we have a simple program built around a popular vector database called Chroma. The file name is vectordb.py. Open the file either by clicking on [**genai/vectordb.py**](./genai/vectordb.py) or by entering the command below in the codespace's terminal.

```
code vectordb.py
```

2. For purposes of not having to load a lot of data and documents, we've *seeded* some random data strings in the file that we're loosely referring to as *documents*. These can be seen in the *datadocs* section of the file.
![data docs](./images/gaidd47.png?raw=true "Data docs")

3. Likewise, we've added some metadata in the way of categories for the data items. These can be seen in the *categories* section.
![data categories](./images/gaidd48.png?raw=true "Data categories")

4. Go ahead and run this program using the command shown below. This will take the document strings, create embeddings and vectors for them in the Chroma database section and then wait for us to enter a query.
```
python vectordb.py
```
![waiting for input](./images/gaidd49.png?raw=true "Waiting for input")

5. You can enter a query here about any topic and the vector database functionality will try to find the most similar matching data that it has. Since we've only given it a set of 10 strings to work from, the results may not be relevant or very good, but represent the best similarity match the system could find based on the query. Go ahead and enter a query. Some sample ones are shown below, but you can choose others if you want. Just remember it will only be able to choose from the data we gave it. The output will show the closest match from the doc strings and also the similarity and category.
```
Tell me about food.
Who is the most famous person?
How can I learn better?
```
![query results](./images/gaidd50.png?raw=true "Query results")

6. After you've entered and run your query, you can add another one or just type *exit* to stop.

7. Now, let's update the number of results that are returned so we can query on multiple topics. In the file *vectordb.py*, change line 70 to say *n_results=3,* instead of *n_results=1,*. Make sure to save your changes afterwards.

![changed number of results](./images/gaidd51.png?raw=true "Changed number of results")

8. Run the program again with *python vectordb.py*. Now you can try more complex queries or try multiple queries (separated by commas). 

![multiple queries](./images/gaidd52.png?raw=true "Multiple queries")
 
9. When done querying the data, if you have more time, you can try modifying or adding to the document strings in the file, then save your changes and run the program again with queries more in-line with the data you provided.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 3 - Working with RAG implemented with vector databases**

**Purpose: In this lab, we’ll build on the use of vector databases to parse a PDF and allow us to include it in context for LLM queries.**

1. In our repository, we have a simple program built for doing basic RAG processing. The file name is rag.py. Open the file either by clicking on [**genai/rag.py**](./genai/rag.py) or by entering the command below in the codespace's terminal.

```
code rag.py
```

2. This program reads in a PDF, parses it into chunks, creates embeddings for the chunks and then stores them in a vector database. It then adds the vector database as additional context for the prompt to the LLM. There is an example pdf named *data.pdf* in the *samples* directory. It contains the same random document strings that were in some of the other programs. You can look at it in the GitHub repo if interested. Open up https://github.com/skillrepos/genai-dd/blob/main/samples/data.pdf if interested.

3. You can now run the program and pass in the ../samples/data.pdf file. This will read in the pdf and tokenize it and store it in the vector database. (Note: A different PDF file can be used, but it needs to be one that is primarily just text. The PDF parsing being used here isn't sophisticated enough to handle images, etc.)
```
python rag.py ../samples/data.pdf
```
![reading in the pdf](./images/gaidd54.png?raw=true "Reading in the PDF")

4. The program will be waiting for a query. Let's ask it for a query about something only in the document. As a suggestion, you can try the one below. The AI will take a while to return results.
```
What does the document say about art and literature topics?
```
5. The response should include only conclusions based off the information in the document.
![results from the doc](./images/gaidd56.png?raw=true "Results from the doc")
  
6. Now, let's ask it a query for some extended information. For example, try the query below. Then hit enter.
```
Give me 5 facts about the Mona Lisa
```
7. In the data.pdf file, there is one (and only one) fact about the Mona Lisa - an obscure one about no eyebrows. In the output, you will probably see this fact near the top as the tools pull the fact from the doc. But the other facts are also based on this one or the LLM telling you couldn't produce any other different facts. 

![5 facts about the Mona Lisa](./images/gaidd55.png?raw=true "5 facts about the Mona Lisa")
   
8. The reason the LLM couldn't add any other facts was due to the PROMPT_TEMPLATE we have in the file. Take a look at it starting around line 29. Note how it limits the LLM to only using the context that comes from our doc (line 51).

![prompt template](./images/rag30.png?raw=true "prompt template")

![doc context](./images/rag31.png?raw=true "doc context")

9. To change this so the LLM can use our context and its own training, we need to change the PROMPT_TEMPLATE. Replace the existing PROMPT_TEMPLATE at lines 29-37 with the lines below. Afterwards, your changes should look like the screenshot below.
```
PROMPT_TEMPLATE = """
Answer the question: {question} using whatever resources you have.
Include any related information from {context} as part of your answer.
Provide a detailed answer.
Don’t justify your answers.
"""
```
![new prompt template](./images/rag32.png?raw=true "new prompt template")

10. **Save your changes** and run the program. Enter the same query "Give me 5 facts about the Mona Lisa". This time, the program will run for several minutes and then the LLM should return 5 "real" facts about the Mona Lisa with our information included. Notice the highlighted part of the fourth item in the screenshot below.  (If the answer isn't returned by the time the break is over, you can just leave it running and check back later.)

```
python rag.py ../samples/data.pdf
```
![new output](./images/rag33.png?raw=true "new output")

<p align="center">
**[END OF LAB]**
</p>
</br></br>


**Lab 4 - Implementing Graph RAG with Neo4j**

**Purpose: In this lab, we'll see how to implement Graph RAG by querying a Neo4j database and using Ollama to generate responses.**

1. For this lab, we'll need a neo4j instance running. We'll use a docker image for this that is already populated with data for us. There is a shell script named [**neo4j/neo4j-setup.sh**](./neo4j/neo4j-setup.sh) that you can run to start the neo4j container running. Change to the neo4j directory and run the script. This will take a few minutes to build and start. Afterwards you can change back to the *genai* subdirectory. Be sure to include the "&" to run this in the background.

```
cd /workspaces/rag/neo4j

./neo4j-setup.sh 1 &

```

2. When done, you may see an "INFO Started" or just a "naming to docker.io/library/neo4j:custom" message. The container should then be running. You can just hit *Enter* and do a *docker ps* command to verify you see a "neo4j:custom" container with "Up # seconds" in the STATUS column.

```
docker ps
```
![container check](./images/rag35.png?raw=true "container check")

3. For the next steps, make sure you're back in the *genai* directory. In here, we have a simple Python program to interact with the graph database and query it. The file name is lab4.py. Open the file either by clicking on [**genai/lab4.py**](./genai/lab4.py) or by entering the *code* command below in the codespace's terminal.

```
cd ../genai
code lab4.py
```

4. You can look around this file to see how it works. It simply connects to the graph database, does a Cypher query (see the function *query_graph* on line 6), and returns the results. For this one, the graph db was initialized with information that *Ada Lovelace, a Mathematician, worked with Alan Turing, a Computer Scientist*.

5. When done looking at the code, go ahead and execute the program using the command below. When it's done, you'll be able to see the closest match from the knowledge base data file to the query.
```
python lab4.py
```
![running lab4 file](./images/rag21.png?raw=true "running lab4 file")

5. Now, let's update the code to pass the retrieved answer to an LLM to expand on. We'll be using the llama3 model that we setup with Ollama previously. For simplicity, the changes are already in a file in [**extra/lab4-changes.txt**](./extra/lab4-changes.txt) To see and merge the differences, we'll use the codespace's built-in diff/merge functionality. Run the command below.

```
code -d /workspaces/rag/extra/lab4-changes.txt /workspaces/rag/genai/lab4.py
```

6. Once you have this screen up, take a look at the added functionality in the *lab4-changes.txt* file. Here we are passing the answer collected from the knowledge base onto the LLM and asking it to expand on it. To merge the changes, you can click on the arrows between the two files (#1 and #2 in the screenshot) and then close the diff window via the X in the upper corner (#3 in the screenshot).

![lab 4 diff](./images/rag22.png?raw=true "lab 4 diff")

7. Now, you can go ahead and run the updated file again to see what the LLM generates using the added context. Note: This will take several minutes to run.  (If you happen to get an error about not being able to establish a connection, your ollama server may not be running any longer.  If that's the case, you can restart it via the command *"ollama serve &"* and then rerun the python command again.)

```
python lab4.py
```

8. After the run is complete, you should see additional data from the LLM related to the additional context with an interesting result!

![lab output 4](./images/rag23.png?raw=true "lab output 4")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 5 - Simplifying RAG with Frameworks and LLMs**

**Purpose: In this lab, we'll see how to simplify Graph RAG by leveraging frameworks and using LLMs to help generate queries.**

1. In our last lab, we hardcoded Cypher queries and worked more directly with the Graph database. Let's see how we can simplify this.

2. First, we need a different graph database. Again, we'll use a docker image for this that is already populated with data for us. Change to the neo4j directory and run the script, but note the different parameter ("2" instead of "1"). This will take a few minutes to build and start. Be sure to add the "&" to run this in the background.

(When it is ready, you may see a "*INFO  [neo4j/########] successfully initialized:*" message or one that says "naming to docker.io/library/neo4j:custom".) Just hit *Enter* and you can change back to the *workspaces/rag* subdirectory. 

```
cd /workspaces/rag/neo4j
./neo4j-setup.sh 2 &
cd ..
``` 

3. This graph database is prepopulated with a large set of nodes and relationships related to movies. This includes actors and directors associated with movies, as well as the movie's genre, imdb rating, etc. You can take a look at the graph nodes by running the following commands in the terminal. **You should be in the "root" directory (/workspaces/rag) when you run these commands.**

```
npm i -g http-server
http-server
```

3. After a moment, you should see a pop-up dialog that you can click on to open a browser to see some of the nodes in the graph. It will take a minute or two to load and then you can zoom in by using your mouse (roll wheel) to see more details.

![running local web server](./images/rag24.png?raw=true "running local web server")
![loading nodes](./images/rag25.png?raw=true "loading nodes")
![graph nodes](./images/rag26.png?raw=true "graph nodes")


4. When done, you can stop the *http-server* process with *Ctrl-C*. Now, let's go back and create a file to use the langchain pieces and the llm to query our graph database. Change back to the *genai* directory and create a new file named lab5.py.
```
cd genai
code lab5.py
```
5. First, add the imports from *langchain* that we need. Put the following lines in the file you just created.
```
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
```
6. Now, let's add the connection to the graph database. Add the following to the file.
```
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jtest",
    enhanced_schema=True,
)
```
7. Next, let's create the chain instance that will allow us to leverage the LLM to help create the Cypher query and help frame the answer so it makes sense. We'll use Ollama and our llama3 model for both the LLM to create the Cypher queries and the LLM to help frame the answers.
```
chain = GraphCypherQAChain.from_llm(
    cypher_llm=Ollama(model="llama3",temperature=0),
    qa_llm=Ollama(model="llama3",temperature=0),
    graph=graph, verbose=True,
)
```

8. Finally, let's add the code loop to take in a query and invoke the chain. After you've added this code, save the file.
```
while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    response = chain.invoke({"query": query})
    print(response["result"])
```

10. Now, run the code.
```
python lab5.py
```
11. You can prompt it with queries related to the info in the graph database, like:
```
Who starred in Star Trek : Generations?
Which movies are comedies?
```
(Ignore the initial error about "NoneType".)

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 6 - Implementing Agentic RAG**

**Purpose: In this lab, we’ll see how to setup an agent using RAG with a tool.**

1. In this lab, we'll download a medical dataset, parse it into a vector database, and create an agent with a tool to help us get answers. First,let's take a look at a dataset of information we'll be using for our RAG context. We'll be using a medical Q&A dataset called [**keivalya/MedQuad-MedicalQnADataset**](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset). You can go to the page for it on HuggingFace.co and view some of it's data or explore it a bit if you want. To get there, either click on the link above in this step or go to HuggingFace.co and search for "keivalya/MedQuad-MedicalQnADataset" and follow the links.
   
![dataset on huggingface](./images/rag27.png?raw=true "dataset on huggingface")    

2. Now, let's create the Python file that will pull the dataset, store it in the vector database and invoke an agent with the tool to use it as RAG. First, create a new file for the project.
```
code lab6.py
```

3. Now, add the imports.
```
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
```

4. Next, we pull and load the dataset.
   
```
data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
data = data.to_pandas()
data = data[0:100]
df_loader = DataFrameLoader(data, page_content_column="Answer")
df_document = df_loader.load()
```

5. Then, we split the text into chunks and load everything into our Chroma vector database.
```
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1250,
                                      separator="\n",
                                      chunk_overlap=100)
texts = text_splitter.split_documents(df_document)

# set some config variables for ChromaDB
CHROMA_DATA_PATH = "vdb_data/"
embeddings = FastEmbedEmbeddings()  

# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(df_document, embeddings, persist_directory=CHROMA_DATA_PATH)
```
6. Set up memory for the chat, and choose the LLM.
```
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=4, #Number of messages stored in memory
    return_messages=True #Must return the messages in the response.
)

llm = Ollama(model="llama3",temperature=0.0)
```

7. Now, define the mechanism to use for the agent and retrieving data. ("qa" = question and answer) 
```
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_chroma.as_retriever()
)
```

8. Define the tool itself (calling the "qa" function we just defined above as the tool).
from langchain.agents import Tool

```
#Defining the list of tool objects to be used by LangChain.
tools = [
   Tool(
       name='Medical KB',
       func=qa.run,
       description=(
           'use this tool when answering medical knowledge queries to get '
           'more information about the topic'
       )
   )
]
```

8. Create the agent using the LangChain project *hwchase17/react-chat*.
```
prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(
   tools=tools,
   llm=llm,
   prompt=prompt,
)

# Create an agent executor by passing in the agent and tools
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               memory=conversational_memory,
                               max_iterations=30,
                               max_execution_time=600,
                               #early_stopping_method='generate',
                               handle_parsing_errors=True
                               )
```

9. Add the input processing loop.
```
while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    agent_executor.invoke({"input": query})
```
10. Now, **save the file** and run the code.
```
python lab6.py
```
11. You can prompt it with queries related to the info in the dataset, like:
```
I have a patient that may have Botulism. How can I confirm the diagnosis?
```

<p align="center">
**[END OF LAB]**
</p>
</br></br>

<p align="center">
**THANKS!**
</p>
