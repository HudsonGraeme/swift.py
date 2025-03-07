from enum import Enum, auto
from typing import Any, Callable, Optional
import select
import socket
import os
import threading
import time


class DispatchSourceType(Enum):
    DATA_ADD = auto()
    DATA_OR = auto()
    MACH_SEND = auto()
    MACH_RECV = auto()
    MEMORY_PRESSURE = auto()
    PROC = auto()
    READ = auto()
    SIGNAL = auto()
    TIMER = auto()
    VNODE = auto()
    WRITE = auto()


class DispatchSource:
    def __init__(
        self, type: DispatchSourceType, handle: Any = None, mask: int = 0, queue=None
    ):
        self.type = type
        self.handle = handle
        self.mask = mask
        self.queue = queue
        self._cancelled = False
        self._event_handler = None
        self._cancel_handler = None
        self._running = False
        self._lock = threading.Lock()
        self._monitor_thread = None

    def set_event_handler(self, handler: Callable):
        with self._lock:
            self._event_handler = handler

    def set_cancel_handler(self, handler: Callable):
        with self._lock:
            self._cancel_handler = handler

    def cancel(self):
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                if self._cancel_handler:
                    if self.queue:
                        self.queue.async_(self._cancel_handler)
                    else:
                        self._cancel_handler()

    def resume(self):
        with self._lock:
            if not self._running and not self._cancelled:
                self._running = True
                self._start_monitoring()

    def suspend(self):
        with self._lock:
            self._running = False

    def _start_monitoring(self):
        if self.type == DispatchSourceType.READ:
            self._monitor_thread = threading.Thread(target=self._monitor_read)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
        elif self.type == DispatchSourceType.WRITE:
            self._monitor_thread = threading.Thread(target=self._monitor_write)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
        elif self.type == DispatchSourceType.TIMER:
            self._monitor_thread = threading.Thread(target=self._monitor_timer)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def _monitor_read(self):
        if not isinstance(self.handle, (int, socket.socket)):
            return

        while self._running and not self._cancelled:
            try:
                r, _, _ = select.select([self.handle], [], [], 0.1)
                if r and self._event_handler:
                    if self.queue:
                        self.queue.async_(self._event_handler)
                    else:
                        self._event_handler()
            except:
                break

    def _monitor_write(self):
        if not isinstance(self.handle, (int, socket.socket)):
            return

        while self._running and not self._cancelled:
            try:
                _, w, _ = select.select([], [self.handle], [], 0.1)
                if w and self._event_handler:
                    if self.queue:
                        self.queue.async_(self._event_handler)
                    else:
                        self._event_handler()
            except:
                break

    def _monitor_timer(self):
        interval = self.handle if isinstance(self.handle, (int, float)) else 1.0
        while self._running and not self._cancelled:
            time.sleep(interval)
            if self._event_handler:
                if self.queue:
                    self.queue.async_(self._event_handler)
                else:
                    self._event_handler()
