"""
Extend pp.Node to be a dask collection.

Using the 'distributed' scheduler, I get
TypeError: ('Could not serialize object of type function', '<function input_node.<locals>.<lambda> at 0x7f1358bfe5e0>')

Using the 'processes' scheduler, I get
TypeError: cannot pickle 'scipp._scipp.core.Variable' object
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
            node: CollectionNode = pending.pop(0)
            # Using {name: n.id for name, n in node.kwparents.items()}
            # as kwargs in the task spec with apply would treat n.id as a literal string
            # and not as a task key.
            task_graph[node._combined_key()] = (
                apply,
                node.func,
                [n._combined_key() for n in node.parents],
                (dict, [[name, n._combined_key()] for name, n in node.kwparents.items()]),
            )
            # We could store the data of input nodes directly in the task graph
            # instead of adding the nodes as a separate task. But this requires
            # detecting input nodes and extracting their data, which does not seem to
            # be easily possible at this point.

            pending.extend(chain(node.parents, node.kwparents.values()))
        return task_graph

    def __dask_keys__(self):
        # Same as dask.Delayed
        return [f'{self.name}-{self.id}']

    def _combined_key(self):
        return ''.join(self.__dask_keys__())

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    def __dask_postcompute__(self):
        return lambda r: r, ()

    def __dask_postpersist__(self):
        raise NotImplementedError("Persist not implemented for CollectionNode")

    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __dask_tokenize__(self):
        # Every time we make a node, it gets a new id.
        # So this is enough to uniquely identify the collection.
        # This would break if nodes were copied and assigned a different function
        # or different parents.
        return self.id

    def compute(self, scheduler=None, optimize_graph=False, **kwargs):
        scheduler = dask.base.get_scheduler(scheduler=scheduler, collections=(self,))
        return scheduler(self.__dask_graph__(), self._combined_key(), **kwargs)

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
    # Set the scheduler, this one is the default.
    dask.config.set(scheduler='threading')

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
