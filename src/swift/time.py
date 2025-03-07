import time
from enum import Enum, auto
from typing import Union, Optional


class DispatchTimeResult(Enum):
    SUCCESS = 0
    TIMED_OUT = auto()


class DispatchTime:
    FOREVER = float("inf")
    NOW = 0

    @staticmethod
    def now() -> float:
        return time.time()

    @staticmethod
    def when(deadline: float) -> float:
        return time.time() + deadline


class DispatchWallTime:
    @staticmethod
    def now() -> float:
        return time.time()

    @staticmethod
    def when(deadline: float) -> float:
        return time.time() + deadline


def dispatch_time(
    when: Union[float, DispatchTime], delta: Optional[float] = None
) -> float:
    if isinstance(when, (int, float)):
        base_time = when
    elif when == DispatchTime.NOW:
        base_time = time.time()
    else:
        base_time = time.time()

    return base_time + (delta or 0)


def dispatch_walltime(
    when: Union[float, DispatchWallTime], delta: Optional[float] = None
) -> float:
    if isinstance(when, (int, float)):
        base_time = when
    else:
        base_time = time.time()

    return base_time + (delta or 0)
