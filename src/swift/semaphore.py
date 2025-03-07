import threading
from typing import Optional


class DispatchSemaphore:
    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Semaphore value must be >= 0")
        self._semaphore = threading.Semaphore(value)

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._semaphore.acquire(timeout=timeout)

    def signal(self) -> int:
        self._semaphore.release()
        return 0
