import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

############################### ORIGINAL VERSION (NO QUEUE, BATCHER, LOAD BALANCER OR AUTOSCALER) ###############################

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to("cuda")

chat_pipeline = pipeline("text-generation", model="facebook/opt-125m", device=0)
# Note: try this 1.5B model if you got enough GPU memory
# chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")



## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    """
    Computes an average-pool embedding for the given text using a pre-trained transformer model.

    Args:
        text (str): The input text for which to compute an embedding.

    Returns:
        np.ndarray: A NumPy array representing the averaged token embeddings of the input text.
    """
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """
    Retrieves the top-k most relevant documents by computing dot-product similarities
    between the query embedding and precomputed document embeddings.

    Args:
        query_emb (np.ndarray): The embedding of the query text.
        k (int, optional): The number of top documents to retrieve. Defaults to 2.

    Returns:
        list: A list of top-k documents (strings) retrieved from the in-memory collection.
    """
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    """
    Implements a simple RAG (Retrieval-Augmented Generation) pipeline:
    1. Embeds the input query.
    2. Retrieves top-k documents.
    3. Constructs a prompt with the query and retrieved documents.
    4. Generates an answer using a text-generation model (LLM).

    Args:
        query (str): The user-provided query or question.
        k (int, optional): The number of documents to retrieve. Defaults to 2.

    Returns:
        str: The generated response from the language model, containing context and answer.
    """
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    """
    Receives a query from the user, calls the RAG pipeline to generate an answer,
    and returns the result as a JSON response.

    Args:
        payload (QueryRequest): A Pydantic model containing the query string and optional number of documents to retrieve.

    Returns:
        dict: A dictionary containing the user's query and the generated result.
    """
    result = rag_pipeline(payload.query, payload.k)
    
    return {
        "query": payload.query,
        "result": result,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
