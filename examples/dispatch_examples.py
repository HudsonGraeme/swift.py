from swift import (
    DispatchQueue,
    DispatchQueueAttributes,
    QualityOfService,
    DispatchWorkItem,
    DispatchWorkItemFlags,
    DispatchSource,
    DispatchSourceType,
    DispatchTime,
    get_main_queue,
    get_global_queue,
    get_global_cpu_queue,
)
import time
import socket
import math
import multiprocessing
from functools import partial


def calculate_primes(n):
    primes = []
    for num in range(2, n):
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            primes.append(num)
    return len(primes)


def calculate_primes_range(start):
    return calculate_primes(50000 + start * 1000)


def fake_network_call(url):
    time.sleep(1)  # Simulate network latency
    return f"Response from {url}"


def print_network_response(url):
    print(fake_network_call(url))


def write_data(shared_data):
    time.sleep(0.1)  # Simulate work
    shared_data.append(len(shared_data))
    print(f"Written: {shared_data[-1]}")


def read_data(shared_data):
    time.sleep(0.1)  # Simulate work
    print(f"Read all: {shared_data}")


def task(n):
    print(f"Starting task {n}")
    time.sleep(0.5)
    print(f"Finished task {n}")


def cpu_bound_example():
    # Using multiprocessing queue for CPU-intensive work
    print("Starting CPU-bound tasks...")
    start = time.time()

    results = []
    global_cpu = get_global_cpu_queue()

    for i in range(4):
        work = DispatchWorkItem(block=calculate_primes_range, args=(i,))
        global_cpu.async_(work)
        results.append(work)

    # Wait for all work items
    for work in results:
        work.wait()

    print(f"CPU-bound tasks completed in {time.time() - start:.2f} seconds")


def io_bound_example():
    # Create a concurrent queue for I/O operations
    io_queue = DispatchQueue(
        "com.example.io",
        attributes=DispatchQueueAttributes.CONCURRENT,
        qos=QualityOfService.UTILITY,
    )

    print("\nStarting I/O-bound tasks...")
    start = time.time()

    urls = [f"https://api.example.com/endpoint{i}" for i in range(5)]
    results = []

    for url in urls:
        work = DispatchWorkItem(block=print_network_response, args=(url,))
        io_queue.async_(work)
        results.append(work)

    for work in results:
        work.wait()

    print(f"I/O-bound tasks completed in {time.time() - start:.2f} seconds")


def barrier_example():
    queue = DispatchQueue(
        "com.example.barrier", attributes=DispatchQueueAttributes.CONCURRENT
    )
    shared_data = []

    print("\nStarting barrier example...")

    # Add some writes
    for _ in range(3):
        work = DispatchWorkItem(block=write_data, args=(shared_data,))
        queue.async_(work)

    # Barrier to ensure all writes complete before reading
    barrier = DispatchWorkItem(
        flags=DispatchWorkItemFlags.BARRIER, block=read_data, args=(shared_data,)
    )
    queue.async_(barrier)

    # More writes after barrier
    for _ in range(2):
        work = DispatchWorkItem(block=write_data, args=(shared_data,))
        queue.async_(work)

    time.sleep(1)  # Wait for completion


def timer_source_example():
    print("\nStarting timer source example...")
    timer_queue = DispatchQueue(
        "com.example.timer", attributes=DispatchQueueAttributes.CONCURRENT
    )

    counter = {"value": 0}
    source = DispatchSource(DispatchSourceType.TIMER, handle=0.5, queue=timer_queue)

    def timer_fired():
        counter["value"] += 1
        print(f"Timer fired: {counter['value']}")
        if counter["value"] >= 5:
            source.cancel()

    source.set_event_handler(timer_fired)
    source.resume()

    time.sleep(3)  # Let it run for a bit


def serial_queue_example():
    print("\nStarting serial queue example...")
    serial_queue = DispatchQueue("com.example.serial")

    for i in range(3):
        work = DispatchWorkItem(block=task, args=(i,))
        serial_queue.async_(work)

    time.sleep(2)  # Wait for completion


if __name__ == "__main__":
    # Required for Windows support
    multiprocessing.freeze_support()

    print("=== Swift.py GCD Examples ===\n")

    cpu_bound_example()
    io_bound_example()
    barrier_example()
    timer_source_example()
    serial_queue_example()

    print("\nAll examples completed!")
