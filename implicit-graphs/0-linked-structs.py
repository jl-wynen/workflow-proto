"""
Graph encoded as refs between nodes.
Nodes are Handles (=Cells, data) and Propagators (computation).
Executed by pulling data from leaves.

Pros:
- Simple
- Executes minimum needed propagators
- Composable (function can just return a leave, using this as an input merges graphs)
- Close to normal Python semantics

Cons:
- Cannot get full graph, only parents.
- Easy to get circular refs.
- Every intermediate handle is always kept alive -> everything is cached
"""

from __future__ import annotations

import uuid
from typing import Callable, Generic, Sequence, TypeVar

import scipp as sc
from graphviz import Digraph


def show_graph(n: Handle) -> str:
    dot = Digraph(strict=True)

    pending = [n]
    visited = set()
    while pending:
        n = pending.pop()
        visited.add(n)
        if isinstance(n, Handle):
            dot.node(n.id, label=n.name, shape="box")
            if (s := n.source()) is not None:
                dot.edge(s.id, n.id)
                if s not in visited:
                    pending.append(s)
        else:
            dot.node(n.id, label=n.name, shape="circle")
            for i in n.inputs:
                dot.edge(i.id, n.id)
                if i not in visited:
                    pending.append(i)

    return str(dot)


T = TypeVar("T")


class Handle(Generic[T]):
    def __init__(self, *, data: T | None, source: Propagator | None, name: str) -> None:
        if data is None and source is None:
            raise ValueError("Either data or source must be provided")
        self._data = data
        self._source = source
        self._name = name
        self._id = str(uuid.uuid4())

    def get(self) -> T:
        # TODO raise if no data?
        #      or return None?
        if self._data is None:
            raise RuntimeError("Cannot get data, handle has none")
        return self._data

    def source(self) -> Propagator | None:
        return self._source

    def compute(self) -> T:
        if self._source is None:
            return self._data
        self._data = self._source()
        return self._data

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def id(self) -> str:
        return self._id

    def __repr__(self) -> str:
        return f"Handle({self._name}, data={self._data!r}, source={self._source!r})"


class Propagator:
    def __init__(self, func: Callable, inputs: Sequence[Handle], *, name: str) -> None:
        if any(not isinstance(inp, Handle) for inp in inputs):
            raise TypeError("All inputs must be handles")

        self._func = func
        self._inputs = tuple(inputs)
        self._name = name
        self._id = str(uuid.uuid4())

    def __call__(self):
        # in prop nets, this would stuff the result directly into a cell
        return self._func(*(inp.compute() for inp in self._inputs))

    @property
    def inputs(self) -> tuple[Handle]:
        return self._inputs

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._id

    def __repr__(self) -> str:
        return f"Propagator({self._name}{self._inputs!r})"


def parameter(name: str, p: T) -> Handle[T]:
    return Handle(data=p, source=None, name=name)


def computed(prop: Propagator) -> Handle:
    return Handle(data=None, source=prop, name=str(uuid.uuid4()))


def make_propagator(func: Callable) -> Callable:
    def impl(*args):
        return computed(Propagator(func=func, inputs=args, name=func.__name__))

    return impl


@make_propagator
def add(a: sc.Variable, b: sc.Variable) -> sc.Variable:
    return a + b


@make_propagator
def div(a: sc.Variable, b: sc.Variable) -> sc.Variable:
    return a / b


def main() -> None:
    a = parameter("a", sc.scalar(2))
    b = parameter("b", sc.scalar(3))
    c = add(a, b)
    c.name = "c"
    d = div(c, a)
    d.name = "d"
    # print(d.compute())
    # print(c.get())

    e = add(c, b)
    e.name = "e"
    # print(e.compute())

    with open("graph.dot", "w") as f:
        f.write(show_graph(d))


if __name__ == "__main__":
    main()
