import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
        return result
    return wrapper


class Timer:
    """
    example: 
    # Example usage of the Timer class
    timer = Timer()

    @timer
    def example_function(n):
        sum = 0
        for i in range(n):
            sum += i
        return sum

    @timer
    def another_function(x):
        time.sleep(x)

    # Running the functions
    example_function(10000)
    example_function(100000)
    another_function(1)
    another_function(2)

    # Retrieve and print all timing data
    print(timer.get_timings())
    """
    def __init__(self):
        self.records = {}

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if func.__name__ not in self.records:
                self.records[func.__name__] = []
            self.records[func.__name__].append(elapsed_time)
            
            print(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
            return result
        return wrapped_func

    def get_timings(self):
        return self.records

