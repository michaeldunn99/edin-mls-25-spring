# Task 2

A modular FastAPI-based LLM-serving system that combines document retrieval with text generation, demonstrating a progression from a baseline impelemntation to a production-style deployment with batching, load balancing, and autoscaling.

## Overview
This project implements an end-to-end LLM-serving system that includes a request queue, batcher, load balancer and autoscaler as per the illustration below:
![system_overview](https://github.com/user-attachments/assets/8ab21228-b570-4570-a66c-bfe0eac1ee8b)

There are three system designs:
1. **Baseline** (no queue, batching, load balancer or autoscaler)
2. **Queued-Batched Design** (no load balancer or autoscaler)
3. **Scaled-Balanced Design** (as per illustration above)

Key components:
- Embedding model: `intfloat/multilingual-e5-large-instruct`
- Generator model: `facebook/opt-125m` (or `Qwen/Qwen2.5-1.5B-Instruct` if resources allow)
- Web API: FastAPI
- Load testing: `tests/test_end_to_end.py`

---
## Environment Setup

```bash
# Step 1: Create a Python environment
conda create -n rag python=3.10 -y
conda activate rag

# Step 2: Clone repository and install requirements
git clone https://github.com/ed-aisys/edin-mls-25-spring.git
cd edin-mls-25-spring/task-2
pip install -r requirements.txt
```

## Running the Baseline Design
Start the Baseline system
```bash
python serving_rag_v0.py
```

Query the Baseline system
```bash
curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "Which animals can hover in the air?"}'
```

## Running the Queued-Batched Design
Start the Queued-Batched system
```bash
python serving_rag_v1.py
```

Query the Queued-Batched system
```bash
curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "Which animals can hover in the air?"}'
```

## Running the Scaled-Balanced Design
**Note:**  
It is important to start the load balancer before the autoscaler, as there system can't start if the autoscaler is started first and tries to register the instances with the load balancer that hasn't been started yet.

Start the load balancer in one terminal. This listens at port ```http://localhost:9000```
```bash
python load_balancer_round_robin.py
```

Start the Autoscaler in a different terminal (defaults to 2 warm instances on ports ```http://localhost:8000``` and ```http://localhost:8001```). This spawns two backend instances of ```serving_rag_v1.py```, scaling as the input load scales.
```bash
python autoscaler.py
```

This system can be tested as described next.

## Testing
Once you've started **one of the three systems** (Baseline, Queued-Batched, or Scaled-Balanced), you can test their performance using the provided script:
```bash
python tests/test_end_to_end.py --mode ideal --target load_balancer_round_robin
```

### CLI Options

| Argument     | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `--mode`     | Type of request arrival. Options: `ideal`, `poisson`.     |
| `--target`   | Endpoints, for testing Baseline or Queued-Batched use original, for testing Scaled-Balanced use load_balancer_round_robin: `original`, `load_balancer_round_robin`. |


**Note:**  
Wait for the backends to spin-up after starting the autoscaler before running the test file. 

