from __future__ import annotations

import uuid
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


class Vertex:
    def __init__(self, *, name: str | None = None) -> None:
        self._id = str(uuid.uuid4())
        self._name = name if name is not None else self._id

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name


class Cell(Vertex, Generic[T]):
    def __init__(
        self, *, data: T | None, name: str | None = None, always_cache: bool = False
    ) -> None:
        super().__init__(name=name)
        self._data = data
        self._always_cache = always_cache

    def get(self) -> T | None:
        return self._data

    def set(self, x: T) -> None:
        self._data = x

    def clear(self) -> None:
        if not self._always_cache:
            self._data = None

    def __repr__(self) -> str:
        return f"Cell[{self._name}]({self._data})"


class Propagator(Vertex):
    def __init__(self, *, func: Callable, name: str | None = None) -> None:
        super().__init__(name=name)
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class Graph:
    def __init__(self) -> None:
        self._cells = {}
        self._propagators = {}
        self._edges = []  # directed (src, dst)
        # Number of handles for each cell
        self._ref_counts = {}

    def cell(self, id: str) -> Cell:
        return self._cells[id]

    def add_cell(self, *, name: str, data: T, always_cache: bool = False) -> Handle[T]:
        cell = Cell(name=name, data=data, always_cache=always_cache)
        id = cell.id
        self._cells[id] = cell
        self._ref_counts[id] = 0
        return self.acquire(id)

    def add_propagator(self, *, name: str, func: Callable) -> str:
        prop = Propagator(name=name, func=func)
        self._propagators[prop.id] = prop
        return prop.id

    def add_edge(self, source: str, dest: str) -> None:
        assert source in self._cells.keys() | self._propagators.keys()
        assert dest in self._cells.keys() | self._propagators.keys()
        self._edges.append((source, dest))

    def acquire(self, id: str) -> Handle:
        self._ref_counts[id] += 1
        return Handle(id, self)

    def release(self, id: str) -> None:
        self._ref_counts[id] -= 1
        if self._ref_counts[id] == 0:
            self._cells[id].clear()

    def merge_from(self, other: Graph) -> None:
        # TODO Do we need to clear `other`?
        self._cells.update(other._cells)
        self._propagators.update(other._propagators)
        for edge in other._edges:
            if edge not in self._edges:
                self._edges.append(edge)
        self._ref_counts.update(other._ref_counts)

    def parents_of(self, id: str) -> Iterable[str]:
        for src, dst in self._edges:
            if id == dst:
                yield src

    def compute(self, out: str):
        if self._cells[out].get() is not None:
            return self._cells[out].get()

        (prop,) = self.parents_of(out)  # there must be exactly 1 propagator
        inputs = self.parents_of(prop)
        res = self._propagators[prop](*(self.compute(i) for i in inputs))
        if self._ref_counts[out] > 0:
            # TODO need to keep result if other propagator also
            #  reads it in same computation
            self._cells[out].set(res)
        return res

    def report(self) -> str:
        return (
            "(\n  "
            + "\n  ".join(
                repr(self._cells[cell_id]) + f" <{self._ref_counts[cell_id]}>"
                for cell_id in self._cells
            )
            + "\n)"
        )


class Handle(Generic[T]):
    def __init__(self, id: str, graph: Graph) -> None:
        # TODO store Cell instead of just id?
        self._id = id
        self._graph = graph

    def __del__(self) -> None:
        self._graph.release(self._id)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._graph.cell(self._id).name

    @name.setter
    def name(self, n: str) -> None:
        self._graph.cell(self._id).name = n

    def get(self) -> T | None:
        return self._graph.cell(self._id).get()

    @property
    def graph(self) -> Graph:
        return self._graph

    @graph.setter
    def graph(self, graph: Graph) -> None:
        self._graph = graph

    def compute(self):
        self._graph.compute(self._id)

    def __repr__(self) -> str:
        return f"Handle[{self.name}]({self.get()})"


def parameter(name: str, p: T) -> Handle[T]:
    g = Graph()
    return g.add_cell(name=name, data=p, always_cache=True)


def propagator(func):
    def impl(x: Handle, *args: Handle) -> Handle:
        g = x.graph
        for arg in args:
            g.merge_from(arg.graph)
            arg.graph = g

        prop_id = g.add_propagator(name=func.__name__, func=func)
        for i in (x, *args):
            g.add_edge(i.id, prop_id)

        out = g.add_cell(name=str(uuid.uuid4()), data=None)
        g.add_edge(prop_id, out.id)
        return out

    return impl


@propagator
def add(a: T, b: T) -> T:
    return a + b


@propagator
def sub(a: T, b: T) -> T:
    return a - b


@propagator
def neg(a: T) -> T:
    return -a


def example2() -> None:
    # graph with anonymous intermediate + use node multiple times
    a = parameter("a", 1)
    b = parameter("b", 3)
    c = add(a, sub(b, a))
    c.name = "c"
    print("before computation:", a._graph.report())
    c.compute()
    print("after computation: ", a._graph.report())
    assert c.get() == b.get()

    # get a handle to the intermediate node (d)
    g = c._graph
    (p,) = g.parents_of(c.id)
    p = g._propagators[p]
    x = [g.acquire(i) for i in g.parents_of(p.id)]
    aa, d = x
    print("aa: ", aa)
    print("d: ", d)
    print(g.report())

    # recompute the content of d
    d.compute()
    print("d =", d.get())


def example1() -> None:
    # 2-layer graph
    # a handle for every node
    a = parameter("a", 1)
    b = parameter("b", 2)
    c = add(a, b)
    c.name = "c"
    d = neg(c)
    d.name = "d"
    print("c: ", c)
    print("d: ", d)
    print("before computation: ", a._graph.report())
    # del c  # this makes the graph not cache the value of c

    d.compute()
    print("after computation: ", a._graph.report())
    print("d: ", d)

    del c  # removes cached value
    print("after removing handle of c: ", a._graph.report())

    del b  # does not remove cached value because b is a parameter
    print("after removing handle of b: ", a._graph.report())


def main() -> None:
    example1()


if __name__ == "__main__":
    main()
