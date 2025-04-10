import time
from collections import deque 
from threading import Lock

class RequestQueue:
    def __init__(self):
        """
        Initializes a new instance of RequestQueue.
        
        Creates a FIFO queue to store requests and initializes a threading lock 
        to synchronize access to the queue.
        """
        self.queue = deque()
        self.lock = Lock()

    def add_request(self, request):
        """
        Adds a request to the queue with the current timestamp.

        Args:
            request: The request object to add to the queue.

        Returns:
            bool: True if the request is successfully added to the queue.
        """
        # Ensuring that only one thread can modify the queue at a time
        with self.lock:
            # Add the request to the queue with the current timestamp
            timestamp = time.time()
            self.queue.append((timestamp, request))
            return True
        
    def get_batch(self, MAX_BATCH_SIZE, MAX_WAIT_TIME, now=None):
        """
        Retrieves a batch of requests subject to one of the following conditions:
        - The queue contains at least MAX_BATCH_SIZE requests.
        - The first request in the queue has waited for at least MAX_WAIT_TIME seconds.

        Args:
            MAX_BATCH_SIZE (int): The maximum number of requests to include in the batch.
            MAX_WAIT_TIME (int): The maximum time (in seconds) to wait before returning a batch.
            now (float, optional): The current timestamp. If None, the system time is used.

        Returns:
            list: A list of (timestamp, request) tuples representing the batch of requests. 
                  Returns an empty list if neither condition is met.
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
    
