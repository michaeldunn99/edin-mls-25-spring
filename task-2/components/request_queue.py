import time
from collections import deque 
from threading import Lock

class RequestQueue:
    def __init__(self, max_size=None):
        self.queue = deque()
        self.max_size = max_size
        self.lock = Lock()

    def add_request(self, request):
        """Adds a request to the queue with the current timestamp."""
        # Ensuring that only one thread can modify the queue at a time
        with self.lock:
            # Check if the queue has reached its maximum size
            if self.max_size and len(self.queue) >= self.max_size:
                # Print a warning message if the queue is full
                print("Warning: Request queue is full. Cannot add new request.")
                return False
            # Otherwise, add the request to the queue with the current timestamp
            timestamp = time.time()
            self.queue.append((request, timestamp))
            return True
        
    def get_batch(self, MAX_BATCH_SIZE, MAX_WAIT_TIME, now=None):
        """
        Return a batch of requests when one of the conditions is met:
        - The batch size reaches MAX_BATCH_SIZE
        - The wait time exceeds MAX_WAIT_TIME

        """
        now = now or time.time() # now paramter is useful for testing
        # Initialize an empty list to store the batch of requests
        batch = []

        # Ensuring that only one thread can read from the queue at a time
        with self.lock:
            # While neither condition is met, keep checking the queue
            while self.queue and len(batch) < MAX_BATCH_SIZE:
                # Get the oldest request from the queue
                oldest_timestamp, _ = self.queue[0]
                # Calculate the waiting time since the oldest request was added
                waiting_time = now - oldest_timestamp

                # Check if the wait time exceeds MAX_WAIT_TIME
                if waiting_time >= MAX_WAIT_TIME:
                    # Add the oldest request to the batch 
                    batch.append(self.queue.popleft())
                else:
                    break

        # Return the batch of requests
        return batch
    
    def __len__(self):
        """Return the current size of the queue."""
        with self.lock:
            return len(self.queue)
