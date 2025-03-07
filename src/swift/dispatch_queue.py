from enum import Enum, auto
from typing import Callable, Optional, Union
import threading
import multiprocessing
import queue
from time import monotonic as _time, sleep
from .work_item import DispatchWorkItem


class QualityOfService(Enum):
    USER_INTERACTIVE = auto()
    USER_INITIATED = auto()
    DEFAULT = auto()
    UTILITY = auto()
    BACKGROUND = auto()


class DispatchQueueAttributes(Enum):
    CONCURRENT = auto()
    CONCURRENT_WITH_MULTIPROCESSING = auto()


def _worker_process(task_queue, result_queue):
    while True:
        try:
            task = task_queue.get()
            if task is None:
                break
            if isinstance(task, tuple):
                func, args, kwargs = task
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))
            else:
                try:
                    task()
                    result_queue.put(("success", None))
                except Exception as e:
                    result_queue.put(("error", str(e)))
        except (EOFError, BrokenPipeError):
            break
        except Exception as e:
            result_queue.put(("error", f"Worker process error: {str(e)}"))


class DispatchQueue:
    def __init__(
        self,
        label: str,
        attributes: Optional[DispatchQueueAttributes] = None,
        qos: QualityOfService = QualityOfService.DEFAULT,
        process_count: Optional[int] = None,
    ):
        self.label = label
        self.concurrent = attributes in (
            DispatchQueueAttributes.CONCURRENT,
            DispatchQueueAttributes.CONCURRENT_WITH_MULTIPROCESSING,
        )
        self.use_multiprocessing = (
            attributes == DispatchQueueAttributes.CONCURRENT_WITH_MULTIPROCESSING
        )
        self.qos = qos
        self._process_count = process_count or (
            multiprocessing.cpu_count() if self.use_multiprocessing else 4
        )

        if self.use_multiprocessing:
            self._task_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            self._processes = []
        else:
            self._task_queue = queue.Queue()
            self._threads = []

        self._lock = threading.Lock() if not self.use_multiprocessing else None
        self._timer_thread = None
        self._delayed_items = []

        if self.concurrent:
            self._start_workers()
            if not self.use_multiprocessing:
                self._start_timer_thread()

    def _worker_thread(self):
        while True:
            try:
                task = self._task_queue.get()
                if task is None:
                    break
                task()
            finally:
                self._task_queue.task_done()

    def _start_workers(self):
        if self.use_multiprocessing:
            for _ in range(self._process_count):
                p = multiprocessing.Process(
                    target=_worker_process,
                    args=(self._task_queue, self._result_queue),
                    daemon=True,
                )
                p.start()
                self._processes.append(p)
        else:
            for _ in range(self._process_count):
                t = threading.Thread(target=self._worker_thread, daemon=True)
                t.start()
                self._threads.append(t)

    def _start_timer_thread(self):
        if not self.use_multiprocessing:
            self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
            self._timer_thread.start()

    def _timer_loop(self):
        while True:
            now = _time()
            with self._lock:
                ready_items = [(t, item) for t, item in self._delayed_items if t <= now]
                self._delayed_items = [
                    (t, item) for t, item in self._delayed_items if t > now
                ]

            for _, item in ready_items:
                if isinstance(item, DispatchWorkItem):
                    self.async_(item.perform)
                else:
                    self.async_(item)

            sleep(0.01)

    def async_(self, work: Union[Callable, DispatchWorkItem]):
        if isinstance(work, DispatchWorkItem):
            if work.cancelled:
                return
            if self.use_multiprocessing:
                self._task_queue.put((work._block, work._args, work._kwargs))

                def wait_for_result():
                    status, result = self._result_queue.get()
                    if status == "error":
                        work._error = RuntimeError(result)
                    else:
                        work._result = result
                    work._completed = True
                    work._completion_event.set()

                threading.Thread(target=wait_for_result, daemon=True).start()
            else:
                self._task_queue.put(work.perform)
        else:
            if self.use_multiprocessing:
                self._task_queue.put((work, (), {}))
            else:
                self._task_queue.put(work)

    def async_after(self, deadline: float, work: Union[Callable, DispatchWorkItem]):
        if self.use_multiprocessing:

            def delayed_work():
                sleep(deadline)
                if isinstance(work, DispatchWorkItem):
                    work.perform()
                else:
                    work()

            self.async_(delayed_work)
        else:
            with self._lock:
                self._delayed_items.append((_time() + deadline, work))

    def sync(self, work: Union[Callable, DispatchWorkItem]):
        if isinstance(work, DispatchWorkItem) and work.cancelled:
            return None

        if self.concurrent:
            if self.use_multiprocessing:
                if isinstance(work, DispatchWorkItem):
                    self._task_queue.put((work._block, work._args, work._kwargs))
                else:
                    self._task_queue.put((work, (), {}))
                status, result = self._result_queue.get()
                if status == "error":
                    raise RuntimeError(result)
                return result
            else:
                event = threading.Event()
                result_container = {"result": None, "error": None}

                def wrapper():
                    try:
                        if isinstance(work, DispatchWorkItem):
                            work.perform()
                        else:
                            result_container["result"] = work()
                    except Exception as e:
                        result_container["error"] = e
                    finally:
                        event.set()

                self._task_queue.put(wrapper)
                event.wait()

                if result_container["error"]:
                    raise result_container["error"]
                return result_container["result"]
        else:
            if isinstance(work, DispatchWorkItem):
                work.perform()
                return None
            return work()

    def __del__(self):
        if self.use_multiprocessing:
            for _ in range(len(self._processes)):
                self._task_queue.put(None)
            for p in self._processes:
                if p.is_alive():
                    p.join(timeout=1)
                    if p.is_alive():
                        p.terminate()
        else:
            for _ in range(len(self._threads)):
                self._task_queue.put(None)


_main_queue = None
_global_queue = None
_global_cpu_queue = None


def get_main_queue():
    global _main_queue
    if _main_queue is None:
        _main_queue = DispatchQueue("com.swift.py.main")
    return _main_queue


def get_global_queue():
    global _global_queue
    if _global_queue is None:
        _global_queue = DispatchQueue(
            "com.swift.py.global", attributes=DispatchQueueAttributes.CONCURRENT
        )
    return _global_queue


def get_global_cpu_queue():
    global _global_cpu_queue
    if _global_cpu_queue is None:
        _global_cpu_queue = DispatchQueue(
            "com.swift.py.global.cpu",
            attributes=DispatchQueueAttributes.CONCURRENT_WITH_MULTIPROCESSING,
        )
    return _global_cpu_queue
