# `swift.py`

A Python implementation of Apple's Grand Central Dispatch (GCD) pattern, bringing efficient concurrent and parallel execution to Python applications (for no good reason).

## Installation

```bash
# Using uv (recommended - faster)
uv pip install swiftpy-gcd

# Using pip
pip install swiftpy-gcd
```

## Usage

```python
from swift import (
    DispatchQueue,
    DispatchQueueAttributes,
    DispatchWorkItem
)

# Create a concurrent queue for I/O operations
io_queue = DispatchQueue(
    label="com.example.io",
    attributes=DispatchQueueAttributes.CONCURRENT
)

# Async execution
io_queue.async_(lambda: print("Async work"))

# With a work item
work = DispatchWorkItem(block=lambda: print("Work item"))
io_queue.async_(work)

# Delayed execution
io_queue.async_after(5.0, lambda: print("Delayed work"))

# CPU-bound parallel processing
cpu_queue = DispatchQueue(
    label="com.example.cpu",
    attributes=DispatchQueueAttributes.CONCURRENT_WITH_MULTIPROCESSING
)

# This will run in true parallel on multiple cores
cpu_queue.async_(lambda: process_large_dataset())

# Synchronous execution
result = cpu_queue.sync(lambda: compute_something())

# Serial queue for ordered operations
serial_queue = DispatchQueue(label="com.example.serial")
serial_queue.async_(lambda: print("First"))
serial_queue.async_(lambda: print("Second"))  # Guaranteed to run after First
```

## Features

- **Dispatch Queues**: Both serial and concurrent execution
- **Multiprocessing Support**: True parallel execution for CPU-bound tasks
- **Work Items**: Cancellable tasks with completion handlers
- **Delayed Execution**: Schedule work to run after a delay
- **Thread Safety**: Built-in synchronization mechanisms
- **Global Queues**: Pre-configured queues for common scenarios
  - Main queue for UI operations
  - Global concurrent queue for I/O
  - Global CPU queue for parallel processing

## Requirements

- Python 3.8+
- No external dependencies

## Performance

- Efficient thread/process pool management
- Smart handling of CPU vs I/O bound tasks
- Low overhead task scheduling
- Automatic worker scaling based on system capabilities

## License

MIT License

```

```
