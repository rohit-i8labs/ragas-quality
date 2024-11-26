import os
os.system("clear")
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from dotenv import load_dotenv,find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datasets import Dataset
from ragas import evaluate
import json
from langchain_chroma import Chroma
from time import sleep
import requests

#--------------------------------
#Setting Up Global Variables
#--------------------------------
BASE_FILE:str = "intermediate_stuff/shimla.txt"
RAW_JSON_FILE:str = "intermediate_stuff/raw.json"
OUTPUT_DATA_JSON_FILE:str = "intermediate_stuff/output.json"
MODEl:str = "gpt-3.5-turbo-1106"
TEMPERATURE:int = 0.4
ALREDY_HAVE_DATASET:bool = False
USE_CHROMA_DB:bool = True
TEXT_SPLITTER_CHUNK_SIZE:int = 1024
TEXT_SPLITTER_CHUNK_OVERLAP:int = 50
PROMPT_TEMPLATE:str = """You are Resume analyzer. Answer questions based on the Resume provided.If you dont have answer to it, just say I dont have answer to that information.

Context: {context}
Question: {input}
"""
# Define the API URL
url = "http://localhost:5000/rag"

#Initialize empty lists
answer = []
contexts = []
test_set = {}
cost:int = 0
total_questions:int = 0
total_tokens:int = 0
prompt_tokens:int = 0
completion_tokens:int = 0

# Retrieve Data from PDF/TXT file and return the chunks
def get_docs():
    print(f"Loading Base Input File: {BASE_FILE}")
    if ".pdf" in BASE_FILE:
        loader = PyPDFLoader(BASE_FILE)  # Load the PDF file
    elif ".txt"in BASE_FILE:    
        loader = TextLoader(BASE_FILE)  # Load the text file
    else:   
        print("Invalid File type to read from.")
        return 0
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP)
    #Chunk The data
    splitDocs = text_splitter.split_documents(docs) 
    return splitDocs

#Create a vector store
def create_vector_store(docs):
    print("Creating Vector Database")
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    if USE_CHROMA_DB:
        print("Using Chorma DB")
        vectorStore = Chroma(
            collection_name="foo",
            embedding_function=embedding,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
        vectorStore.add_documents(documents=docs)
    else:  
        print("Using In Memory DB")  
        vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

# Load the JSON set and convert to dataset
with open(RAW_JSON_FILE, 'r') as file:
    test_set = json.load(file)
    dataset = Dataset.from_dict(test_set) # Convert sample dataset to Dataset object

# Create a chian of docc,vectorDB and Template
def create_chain(retriever):
    print("Creating Retrieval Chain")
    model = ChatOpenAI(temperature=TEMPERATURE,model=MODEl)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template,
    )

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

#Invoke the chain and ask the question
def ask_question(chain, question):
    global cost,total_questions,total_tokens,prompt_tokens,completion_tokens
    # Prepare the conversation history for the prompt
    # history_text = "\n".join(chat_history)
    
    with get_openai_callback() as cb:
        response = chain.invoke({
            "input": question
            # "history": history_text  # Pass the chat history to the model
        })
    
    print(f"{total_questions+1} -> Total Tokens: {cb.total_tokens}")
    print(f"{total_questions+1} -> Prompt Tokens: {cb.prompt_tokens}")
    print(f"{total_questions+1} -> Completion Tokens: {cb.completion_tokens}")
    print(f"{total_questions+1} -> Total Cost (USD): ${cb.total_cost:.4f}")    
    print("--------------------------------------") 
   
    # Compute Costing Per Question
    cost += cb.total_cost
    total_questions +=1
    total_tokens += cb.total_tokens
    prompt_tokens += cb.prompt_tokens
    completion_tokens += cb.completion_tokens
    
    # Store the question and response in chat history
    # chat_history.append(f"Human: {question}")
    # chat_history.append(f"Assistant: {response}")
    return response    

# Load documents and create vector store and chain
if not ALREDY_HAVE_DATASET:
    docs = get_docs()
    vectorStore = create_vector_store(docs)
    retriever = vectorStore.as_retriever()
    chain = create_chain(retriever=retriever)

    #Recursively Answer all the questions in the list
    for question in test_set['question']:
        response = ask_question(chain, question)
        # contxt = [docs.page_content for docs in retriever.get_relevant_documents(question)]
        context = [doc.page_content for doc in response["context"]]
        answer.append(response['answer'])
        contexts.append(context)
        sleep(1)

    #Append all the answers & contexts to the final dataset.
    test_set['answer']= answer    
    test_set['contexts']= contexts  
    json_data = json.dumps(test_set, indent=2) # Convert dict to json str

    # Write the JSON string to a file
    with open(OUTPUT_DATA_JSON_FILE, "w") as json_file:
        json.dump(test_set, json_file, indent=2)
else:
    print("Dataset Alredy Exists")
    print(f"Reading from Dataset: {OUTPUT_DATA_JSON_FILE}")
    with open(OUTPUT_DATA_JSON_FILE, 'r') as file:
        test_set = json.load(file)

# Send a POST request to the API with the data
print("Evaluating The RAG. Please Wait.")
test_set['cost'] = round(cost,4)
test_set['queries'] = total_questions
response = requests.post(url, json=test_set)

# Print the response status and content
if response.status_code == 200:
    print("Success! Received response:")
    print(response.json())  # Assuming the response is in JSON format
    print("\n--------------------------------------") 
    print(f"Total Queries: {total_questions}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Cost (USD): ${cost:.4f}")    
    print("--------------------------------------") 
else:
    print(f"Failed with status code {response.status_code}")
    print(response.text)