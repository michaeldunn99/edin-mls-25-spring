import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
from pydantic import BaseModel, PrivateAttr
from threading import Thread
from typing import Optional
from queue import Queue
import os
import time
from request_queue import RequestQueue
import uvicorn
import uuid

################################ VERSION WITH REQUEST QUEUE AND BATCHER ################################

app = FastAPI()

# Constants for batching
MAX_BATCH_SIZE = 10
MAX_WAIT_TIME = 0.5  # seconds

# Global request queue
request_queue = RequestQueue()
response_queues = {}

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to("cuda")

# Load text generation model
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m", device=0)

# Compute average-pool embeddings
def get_embedding_batch(texts: list[str]) -> np.ndarray:
    inputs = embed_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding_batch([doc]) for doc in documents])

# Retrieve top-k documents via dot-product
def retrieve_top_k_batch(query_embs: np.ndarray, k_list: list[int]) -> list[list[str]]:
    batch_results = []
    for emb, k in zip(query_embs, k_list):
        sims = doc_embeddings @ emb.T
        top_k_indices = np.argsort(sims.ravel())[::-1][:k]
        batch_results.append([documents[i] for i in top_k_indices])
    return batch_results

# Request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    # Internal private attribute (not part of the schema or exposed to user)
    _id: Optional[str] = PrivateAttr(default=None)

# Background worker thread for batching requests
def batch_worker():
    while True:
        batch = request_queue.get_batch(MAX_BATCH_SIZE, MAX_WAIT_TIME)
        if not batch:
            # If there is no batch, sleep for a short time before checking again
            time.sleep(0.01)
            continue

        try:
            # Extract the queries, ks, and ids from the batch
            requests = [item[1] for item in batch]
            queries = [req.query for req in requests]
            ks = [req.k for req in requests]
            ids = [req._id for req in requests]

            # Get the embeddings for the queries and retrieve the top-k documents
            query_embs = get_embedding_batch(queries)

            retrieved_docs_batch = retrieve_top_k_batch(query_embs, ks)

            # Create prompts for the chat model
            prompts = [
                f"Question: {query}\nContext:\n{chr(10).join(docs)}\nAnswer:"
                for query, docs in zip(queries, retrieved_docs_batch)
            ]

            # Generate responses
            generations = chat_pipeline(prompts, max_length=50, do_sample=True)

            results = [g[0]["generated_text"] for g in generations]

            for req_id, result in zip(ids, results):
                response_queues[req_id].put(result)

        except Exception as e:
            print(f"[Batch Worker ERROR] Chat generation failed: {e}")
            for req_id in ids:
                response_queues[req_id].put(f"Generation failed: {str(e)}")


# Launch background batch processing thread
Thread(target=batch_worker, daemon=True).start()

@app.post("/rag")
def predict(payload: QueryRequest):
    # Generate a unique request ID for the payload.
    payload._id = f"req_{uuid.uuid4()}"  # Set internal ID here

    resp_q = Queue()
    response_queues[payload._id] = resp_q

    # Add the request to the request queue
    request_queue.add_request(payload)

    try:
        result = resp_q.get()  # seconds
    except Exception as e:
        print(f"[{os.environ.get('PORT')}] Timeout or error in response queue: {e}")
        return {"error": "Timeout waiting for batch response"}
    del response_queues[payload._id]

    return {
        "query": payload.query,
        "result": result,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)