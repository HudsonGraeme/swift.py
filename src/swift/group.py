from typing import Callable, Optional
import threading


class DispatchGroup:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._notify_queue = []

    def enter(self):
        with self._lock:
            self._count += 1

    def leave(self):
        with self._lock:
            self._count -= 1
            if self._count == 0:
                self._condition.notify_all()
                for callback in self._notify_queue:
                    callback()
                self._notify_queue.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            if self._count == 0:
                return True
            return self._condition.wait(timeout=timeout)

    def notify(self, queue, execute: Callable):
        with self._lock:
            if self._count == 0:
                queue.async_(execute)
            else:
                self._notify_queue.append(lambda: queue.async_(execute))
