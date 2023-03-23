"""
Define multiple inputs via dask.bag and run a workflow defined as delayed.

Tested with the Jupyter dashboard and it runs the workflow in parallel.

Notes

inputs.map(make_workflow).compute() produces a list of delayed objects.
It passes concrete inputs to the workflow and construct a task graph for each.
But they are not executed.
=> Need to manually compute them in the map call to run them in parallel.

This makes it impossible to inspect the graph as a whole for, e.g., provenance!
"""

from dask.distributed import Client
import dask
import dask.bag as db


@dask.delayed
def foo(a, b):
    return a + b


@dask.delayed
def bar(a, b):
    return a * b


def make_workflow(a):
    b = 3
    c = foo(a, b)
    d = bar(c, a)
    return d


def main() -> None:
    client = Client()
    inputs = db.from_sequence([1, 2])

    w = db.from_sequence(inputs.map(
        lambda a: make_workflow(a).compute()
    ))
    r = w.compute()
    print('result: ', r)


if __name__ == '__main__':
    main()
