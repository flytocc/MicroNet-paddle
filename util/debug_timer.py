""" https://github.com/flytocc/debug-timer
"""

import time
from collections import defaultdict
from typing import Callable


class Timer(object):
    """
    A timer which computes the time elapsed since the tic/toc of the timer.
    """

    def __init__(self):
        super(Timer, self).__init__()
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        return self.average_time if average else self.diff

    def reset(self):
        """
        Reset the timer.
        """
        self.start_time = 0.
        self.diff = 0.
        self.total_time = 0.
        self.average_time = 0.
        self.calls = 0


class _DebugTimer(object):
    """
    Track vital debug statistics.
    Usage:
        1. from timer import debug_timer

        2. with debug_timer("timer1"):
               code1

           debug_timer.tic("timer2")
           code2
           debug_timer.toc("timer2")

           debug_timer.timer3_tic()
           code3
           debug_timer.timer3_toc()

           @debug_timer("timer4")
           def func(*args, **kwargs):
               code4

        3. debug_timer.log()
    """

    __TIMER__ = None
    prefix = ""

    def __new__(cls, *args, **kwargs):
        if cls.__TIMER__ is None:
            cls.__TIMER__ = super(_DebugTimer, cls).__new__(cls)
        return cls.__TIMER__

    def __init__(self, num_warmup=0):
        super(_DebugTimer, self).__init__()
        self.num_warmup = num_warmup
        self.timers = defaultdict(Timer)
        self.context_stacks = []
        self.calls = 0
        self.set_wait(lambda: None)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name.endswith("_tic"):
            return lambda : self.tic(name[:-4])
        elif name.endswith("_toc"):
            return lambda : self.toc(name[:-4])
        raise AttributeError(name)

    def __call__(self, name_or_func):
        if isinstance(name_or_func, str):
            self.context_stacks.append(name_or_func)
            return self

        name = self.context_stacks.pop(-1)
        def func_wrapper(*args, **kwargs):
            with self(name):
                return name_or_func(*args, **kwargs)
        return func_wrapper

    def __enter__(self):
        name = self.context_stacks[-1]
        self.tic(name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        name = self.context_stacks.pop(-1)
        self.toc(name)
        if exc_type is not None:
            raise exc_value

    def set_wait(self, func):
        assert isinstance(func, Callable)
        self.wait = func

    def reset_timer(self):
        for timer in self.timers.values():
            timer.reset()

    def tic(self, name):
        timer = self.timers[name]
        self.wait()
        timer.tic()
        return timer

    def toc(self, name):
        timer = self.timers.get(name, None)
        if timer is None:
            raise ValueError(f"Trying to toc a non-existent Timer which is named '{name}'!")
        if self.calls >= self.num_warmup:
            self.wait()
            return timer.toc(average=False)

    def log(self, logperiod=10, prefix="", log_func=None):
        """
        Log the tracked statistics.
        Eg.: | timer1: xxxs | timer2: xxxms | timer3: xxxms |
        """
        self.calls += 1
        if self.calls % logperiod == 0 and self.timers:
            log_func = log_func or print

            lines = [prefix or self.prefix]
            for name, timer in self.timers.items():
                avg_time = timer.average_time
                suffix = "s"
                if avg_time < 0.01:
                    avg_time *= 1000
                    suffix = "ms"
                lines.append(" {}: {:.3f}{} ".format(name, avg_time, suffix))
            lines.append("")
            log_func("|".join(lines))


debug_timer = _DebugTimer()


if __name__ == "__main__":
    @debug_timer('timer0')
    def code():
        s = 0
        for i in range(100000):
            s += i
        return True

    for _ in range(1000):
        with debug_timer('timer1'):
            code()

        debug_timer.tic("timer2")
        code()
        debug_timer.toc("timer2")

        debug_timer.timer3_tic()
        code()
        debug_timer.timer3_toc()

        debug_timer.log()
