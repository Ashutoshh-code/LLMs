from openai import AzureOpenAI
from datetime import datetime
import hashlib    #It's used to generate hashes — fixed-length strings derived from input dat
import re         #Regular expressions (regex) are patterns used to match and manipulate strings — like finding, validating, or replacing parts of text.
import os
import dotenv

from tqdm import tqdm     #tqdm is a Python library for showing progress bars in loops. ex : 100%|████████████████████| 100/100 [00:01<00:00, 98.76it/s]
import numpy as np
from torch import nn      #torch.nn stands for Neural Networks. It provides all the building blocks to create deep learning models — like layers, loss functions, and activation functions.
import time
import logging    #logging module, which provides a flexible framework for writing log messages from your code.Record warnings, errors, or info for debugging or auditing, Track the flow of program and avoid using print() in production code
from pinecone import Pinecone, ServerlessSpec #This is a configuration helper class used to define serverless index settings, like: Cloud provider (e.g., "aws"), Region (e.g., "us-west-2")
from sentence_transformers import SentenceTransformer, CrossEncoder #CrossEncoder is to instantiate re-ranking system.

logger = logging.getLogger()  #Returns the root logger (or you can pass a name to get a specific logger). It's the central object used to log messages.
logger.setLevel(logging.CRITICAL) #Sets the minimum severity level of messages that the logger will handle.(Max 50), So only messages logged as CRITICAL will be shown; lower levels like ERROR, WARNING, INFO, or DEBUG will be ignored.


import chromadb
from chromadb.config import Settings



from dotenv import dotenv_values  #dotenv_values loads environment variables from a .env file into a dictionary, allowing you to access them in your code.
#specify env name
env_name = '.env'  #This is the name of the file where environment variables are stored. It should be in the same directory as your script.
config = dotenv_values(env_name)  #Loads the environment variables from the specified .env file into a dictionary called config.
OPENAI_API_KEY = config['openai_key']
OPENAI_API_ENDPOINT = config['openai_endpoint']
OPENAI_API_VERSION = config['openai_api_version'] # at the time of authoring, the api version is 2024-02-01
#COMPLETIONS_MODEL_DEPLOYMENT_NAME = config['openai_completions_deployment']
EMBEDDING_MODEL_DEPLOYMENT_NAME = config['openai_embeddings_model']


client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_ENDPOINT,
    api_version=OPENAI_API_VERSION,   #completions_deployment=COMPLETIONS_MODEL_DEPLOYMENT_NAME,
)

ENGINE = config['openai_embeddings_model'] #has vector size 1536


# Helper functions to get lists of vector embeddings from the OpenAI API
def get_embeddings(texts, engine = ENGINE):       # engine = ENGINE is the default value for the engine parameter, so this function can run with only parameter "texts" only.
  response = client.embeddings.create(input = texts, model = engine)
  return [d.embedding for d in list(response.data)]

def get_embedding(text, engine = ENGINE):
  return get_embeddings([text], engine)[0]

print(len(get_embedding("Hello")), len(get_embeddings(["Hello", "World"])))

dimensions = 1536 #This is the dimensionality of the embeddings returned by the OpenAI API for the specified model. It indicates that each text will be represented as a vector with 1536 elements.

  #This will print the length of the first embedding vector returned by get_embeddings for the texts "Hello" and "World".

'''
a) For function get_embeddings(texts, engine = ENGINE), the purpose is this function gets embeddings (vector representations) for a list of texts. Calls the OpenAI API to generate embeddings for the texts. client is assumed to be an OpenAI-client.
b) response.data is a list of objects (each corresponding to one input text). Each object has an embedding attribute (a list of floats). This line extracts each embedding into a list.
c) For function get_embedding(text, engine = ENGINE), the purpose is to get the embedding for a single text. It calls get_embeddings with a list containing just that text and returns the first (and only) embedding from the result.
'''

# ==========+ ChromaDB Setup +==========
chroma_client = chromadb.Client(Settings(
  persist_directory="./chroma_db",
  anonymized_telemetry=False 
))

COLLECTION_NAME = "semantic_search_Chroma"  # Name of the collection in ChromaDB
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# ==========+ Add documents to Chroma +==========
texts = [
    "I love travelling.",
    "AI is transforming the world.",
    "PUBG is the best video game.",
    "Dhoni is one of the best cricketers in the world.",
    "Will AI take over the world?",
    "The weather is nice today.",
    "I enjoy reading books on AI and technology.",
    "The fiture of LLMs is promising.",
    "Python is a versatile programming language.",
    "The future of AI is bright.",
    "I am learning about semantic search.",
    "ChromaDB is a great tool for semantic search.",
    "I love playing cricket.",
    "The future of technology is exciting.",
    "I enjoy watching movies and series.",
    "The advancements in AI are remarkable.",
    "I am interested in machine learning.",
    "The future of gaming is evolving.",
    "I like to explore new places.",
    "The future of programming is evolving.",
    "I am passionate about data science."
]
ids = [f"doc_{i}" for i in range(len(texts))]  # Generate unique IDs for each text
embeddings = get_embeddings(texts)  # Get embeddings for the texts

collection.add(
  ids = ids,
  embeddings = embeddings,
  documents = texts,
  metadatas = [{"category": "example"} for _ in texts]  # Example metadata
)

# ==========+ Querying ChromaDB +==========
query_text = input("Query : ")
query_embedding = get_embedding(query_text)  # Get embedding for the query text

results = collection.query(
    query_embeddings=[query_embedding],  # Use the query embedding
    n_results=2,  # Number of results to return
    include=["documents", "distances"]
)

print("Query Results:")
for doc, score in zip(results["documents"][0], results["distances"][0]):      #Score shoes Euclidean distance, lower scores indicate more similar documents.
    # Print each document and its score
    print(f"Document: {doc}, Score: {score:.4f}") 
    

