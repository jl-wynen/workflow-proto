"""
Build a graph with Plopp and execute it with a scheduler from Dask.
"""

from itertools import chain

import dask
import dask.threaded
import plopp as pp
import scipp as sc
from dask.utils import apply
from rich import print


def add(a, b):
    print("ADD", a, b)
    return a + b


def mul(a, b):
    print("MUL", a, b)
    return a * b


def compute(node):
    target_node = node
    task_graph = {}
    pending = [node]
    while pending:
        node = pending.pop(0)
        # Using {name: n.id for name, n in node.kwparents.items()}
        # as kwargs in the task spec with apply would treat n.id as a literal string
        # and not as a task key.
        task_graph[node.id] = (
            apply,
            node.func,
            [n.id for n in node.parents],
            (dict, [[name, n.id] for name, n in node.kwparents.items()]),
        )
        # We could store the data of input nodes directly in the task graph
        # instead of adding the nodes as a separate task. But this requires
        # detecting input nodes and extracting their data, which does not seem to
        # be easily possible at this point.

        pending.extend(chain(node.parents, node.kwparents.values()))
    print(task_graph)
    return dask.threaded.get(task_graph, target_node.id)


def main() -> None:
    a = sc.scalar(2)
    b = sc.scalar(3)

    an = pp.input_node(a)
    an.name = "a"
    bn = pp.input_node(b)
    bn.name = "b"

    cn = pp.node(add)(an, b=bn)
    cn.name = "c"

    dn = pp.node(mul)(a=cn, b=bn)
    dn.name = "d"

    print(compute(dn))


if __name__ == "__main__":
    main()
