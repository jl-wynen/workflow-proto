"""
Testing logging with dask.delayed

Works fine with the threaded scheduler.

When using distributed.Client, log messages below WARNING are swallowed up somewhere.
Even a custom handler does not see those messages.
Print's still work. But I seem to remember them not showing up in a notebook.
"""

import dask
from dask.distributed import Client
import logging


class ListHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def report(self):
        print("\n".join(f"{record.levelname}: {record.getMessage()}" for record in self.records))


def get_logger():
    return logging.getLogger("dask-log")


def configure_logging():
    hdlr = logging.StreamHandler()
    hdlr.setLevel('INFO')

    lhdlr = ListHandler()
    lhdlr.setLevel('INFO')

    get_logger().setLevel('INFO')
    get_logger().addHandler(hdlr)
    get_logger().addHandler(lhdlr)

    # logging.getLogger().setLevel('INFO')
    # logging.getLogger().addHandler(hdlr)

    return lhdlr


def task(f):
    @dask.delayed
    def impl(*args, **kwargs):
        print(f'PRINT {f.__name__}')
        get_logger().info('INFO %s', f.__name__)
        get_logger().warning('WARN %s', f.__name__)
        return f(*args, **kwargs)

    return impl


@task
def foo(a, b):
    return a + b


@task
def bar(a, b):
    return a * b


def main() -> None:
    client = Client()
    lhdlr = configure_logging()

    a = 1
    b = 2
    c = foo(a, b)
    d = bar(c, a)

    print("graph done:", d)
    dd = d.compute()
    print("result:", dd)

    print("logs: ")
    lhdlr.report()


if __name__ == '__main__':
    main()
