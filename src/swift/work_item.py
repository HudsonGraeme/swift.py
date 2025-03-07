from enum import Flag, auto
from typing import Callable, Optional, Any, Tuple
import threading
import queue
import multiprocessing


class DispatchWorkItemFlags(Flag):
    NONE = 0
    ASSIGN_CURRENT_CONTEXT = auto()
    DETACHED = auto()
    ENFORCE_QOS = auto()
    INHERIT_QOS = auto()
    BARRIER = auto()


class DispatchWorkItem:
    def __init__(
        self,
        flags: DispatchWorkItemFlags = DispatchWorkItemFlags.NONE,
        block: Optional[Callable] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[dict] = None,
    ):
        self.flags = flags
        self._block = block
        self._args = args or ()
        self._kwargs = kwargs or {}
        self._cancelled = False
        self._completed = False
        self._executing = False
        self._lock = threading.Lock()
        self._completion_handlers = []
        self._completion_event = threading.Event()
        self._result = None
        self._error = None

    def perform(self):
        with self._lock:
            if self._cancelled or self._completed:
                return
            self._executing = True

        try:
            if self._block:
                self._result = self._block(*self._args, **self._kwargs)
        except Exception as e:
            self._error = e
        finally:
            with self._lock:
                self._executing = False
                self._completed = True
                handlers = self._completion_handlers.copy()

            self._completion_event.set()
            for handler in handlers:
                handler()

    def cancel(self):
        with self._lock:
            if not self._completed and not self._executing:
                self._cancelled = True

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._completed:
            if self._error:
                raise self._error
            return True

        success = self._completion_event.wait(timeout=timeout)
        if success and self._error:
            raise self._error
        return success

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def executing(self) -> bool:
        with self._lock:
            return self._executing

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def result(self):
        if not self._completed:
            raise RuntimeError("Work item not completed")
        if self._error:
            raise self._error
        return self._result
