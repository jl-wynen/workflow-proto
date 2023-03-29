"""
Extend pp.Node to be a dask collection.
"""

from itertools import chain
from functools import partial

import dask.threaded
import plopp as pp
import scipp as sc
from dask.utils import apply
from rich import print


class CollectionNode(pp.Node):
    def __dask_graph__(self):
        task_graph = {}
        pending = [self]
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
        return task_graph

    def __dask_keys__(self):
        # Is this correct?
        return [self.id]

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    def __dask_postcompute__(self):
        return lambda r: r, ()

    def __dask_postpersist__(self):
        raise NotImplementedError("Persist not implemented for CollectionNode")

    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __dask_tokenize__(self):
        # Or should this be the hash of the result?
        return self.id

    def compute(self, scheduler=None, optimize_graph=False, **kwargs):
        # Should check dask's global scheduler first.
        if scheduler is None:
            scheduler = CollectionNode.__dask_scheduler__
        return scheduler(self.__dask_graph__(), self.id, **kwargs)

    def persist(self, **kwargs):
        raise NotImplementedError("Persist not implemented for CollectionNode")

    def visualize(self, **kwargs):
        raise NotImplementedError("Visualize not implemented for CollectionNode")


def node(func, *args, **kwargs):
    partialized = partial(func, *args, **kwargs)

    def make_node(*args, **kwargs):
        return CollectionNode(partialized, *args, **kwargs)

    return make_node


def input_node(obj):
    return CollectionNode(lambda: obj)


def add(a, b):
    print("ADD", a, b)
    return a + b


def mul(a, b):
    print("MUL", a, b)
    return a * b


def main() -> None:
    a = sc.scalar(2)
    b = sc.scalar(3)

    an = input_node(a)
    an.name = "a"
    bn = input_node(b)
    bn.name = "b"

    cn = node(add)(an, b=bn)
    cn.name = "c"

    dn = node(mul)(a=cn, b=bn)
    dn.name = "d"

    print(dn.compute())


if __name__ == "__main__":
    main()
