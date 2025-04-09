import time
from collections import deque 
from threading import Lock

class RequestQueue:
    def __init__(self):
        # FIFO queue to store requests
        self.queue = deque()
        self.lock = Lock()

    def add_request(self, request):
        """Adds a request to the queue with the current timestamp."""
        # Ensuring that only one thread can modify the queue at a time
        with self.lock:
            # Add the request to the queue with the current timestamp
            timestamp = time.time()
            self.queue.append((timestamp, request))
            return True
        
    def get_batch(self, MAX_BATCH_SIZE, MAX_WAIT_TIME, now=None):
        """
        Return a batch of requests when one of the conditions is met:
        - The batch size reaches MAX_BATCH_SIZE
        - The wait time exceeds MAX_WAIT_TIME

        """
        now = now or time.time()
        batch = []

        with self.lock:
            if not self.queue:
                return batch

            # Condition 1: Enough requests to fill a batch (immediate return)
            if len(self.queue) >= MAX_BATCH_SIZE:
                for _ in range(MAX_BATCH_SIZE):
                    batch.append(self.queue.popleft())
                return batch

            # Condition 2: First request has waited long enough
            oldest_timestamp, _ = self.queue[0]
            if (now - oldest_timestamp) >= MAX_WAIT_TIME:
                while self.queue and len(batch) < MAX_BATCH_SIZE:
                    batch.append(self.queue.popleft())
                return batch

        return batch  # Neither condition met; return empty. Empty batch is handled in batch_worker in serving_rag_v1.py
    
