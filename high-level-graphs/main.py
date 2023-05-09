from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator, Literal, Protocol
import uuid
from dask.utils import apply as dask_apply
import dask
from rich import print
from itertools import chain


"""
    @node(params=("p1", "p2"))
    def foo(a, p1, b="x", p2="y"): ...

    foo(a=a_node)
    foo.build({"p1": "p"})

Arg types:
 - a: parent
 - b: optional given arg
 - p1: parameter
 - p2: optional parameter
 
Parents and given args must be specified in call to foo.
(Optional) parameters must be specified in build.
"""


def reject_extra_args(
    func_name: str,
    *,
    args: tuple[str],
    given_args: dict[str, Any],
    designated_params: set[str],
) -> None:
    for arg in given_args:
        if arg not in args:
            raise TypeError(f"Invalid direct argument for {func_name}: {arg}")
    for arg in designated_params:
        if arg not in args:
            raise TypeError(f"Invalid parameter for {func_name}: {arg}")


def expect_all_args_are_specified(
    func_name: str,
    *,
    args: tuple[str],
    optional: set[str],
    given_args: dict[str, Any],
    designated_params: set[str],
) -> None:
    for arg in args:
        if arg not in optional and arg not in given_args.keys() | designated_params:
            raise TypeError(f"{func_name}() missing argument {arg}")


def validate_args(
    func_name: str,
    *,
    args: tuple[str],
    optional: set[str],
    given_args: dict[str, Any],
    designated_params: set[str],
) -> None:
    reject_extra_args(
        func_name, args=args, given_args=given_args, designated_params=designated_params
    )
    expect_all_args_are_specified(
        func_name,
        args=args,
        optional=optional,
        given_args=given_args,
        designated_params=designated_params,
    )


@dataclass()
class NodeArgs:
    parents: dict[str, Node]
    args: dict[str, Any]
    param_names: set[str]
    params: dict[str, Any]
    optional: set[str]


def classify_args(
    func: Callable, *, given_args: dict[str, Any], designated_params: set[str]
) -> NodeArgs:
    func = getattr(func, "__node_wrapped__", func)

    argspec = inspect.getfullargspec(func)

    args = tuple(argspec.args + argspec.kwonlyargs)
    optional = get_optional_args(argspec)
    validate_args(
        func.__name__,
        args=args,
        optional=optional,
        given_args=given_args,
        designated_params=designated_params,
    )

    parents = {
        k: v
        for k, v in given_args.items()
        if isinstance(v, Node) and k not in designated_params
    }
    other_given_args = {
        k: v
        for k, v in given_args.items()
        if not isinstance(v, Node) and k not in designated_params
    }
    params = {k: v for k, v in other_given_args.items() if k in designated_params}

    return NodeArgs(
        parents=parents,
        args=other_given_args,
        param_names=set(designated_params),
        params=params,
        optional=optional,
    )


class Program(Protocol):
    def compute(self) -> Any:
        ...


class DaskProgram:
    def __init__(self, task_graph: dict, root_name: str) -> None:
        self._task_graph = task_graph
        self._root_name = root_name

    def compute(self) -> Any:
        return dask.get(self._task_graph, self._root_name)


class Node:
    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        func_kwargs: dict[str, Any],
        params: tuple[str],
    ) -> None:
        self._name = name if name is not None else func.__name__
        self._id = None
        self._set_new_id()

        self._func = func
        args = classify_args(
            func, given_args=func_kwargs, designated_params=set(params)
        )
        self._parents = args.parents
        self._args = args.args
        self._param_names = args.param_names
        self._params = args.params
        self._optional = args.optional

    def clone(self) -> Node:
        clone = deepcopy(self)
        for node in clone._depth_first():
            node._set_new_id()
        return clone

    def _set_new_id(self) -> None:
        self._id = self._name + "-" + uuid.uuid4().hex

    def _depth_first(self) -> Generator[Node, None, None]:
        pending = [self]
        visited = set()
        while pending:
            node = pending.pop(0)
            if node._id in visited:
                raise RuntimeError("Not a tree")
            visited.add(node._id)
            yield node
            pending.extend(node._parents.values())

    def _depth_first_with_params(
        self, params: dict
    ) -> Generator[(Node, dict), None, None]:
        pending = [(self, params)]
        visited = set()
        while pending:
            node, params = pending.pop(0)
            if node._id in visited:
                raise RuntimeError("Not a tree")
            visited.add(node._id)
            yield node, {k: v for k, v in params.items() if k not in node._parents}
            pending.extend(
                (parent, params.get(name, {})) for name, parent in node._parents.items()
            )

    def set_param(self, name: str, value: Any) -> None:
        self._params[name] = value

    @property
    def parents(self) -> Generator[tuple[str, Node], None, None]:
        yield from self._parents.items()

    def build(self, params: dict[str, Any], engine: Literal["dask"]) -> Program:
        if engine == "dask":
            return self.build_dask(params)
        raise NotImplementedError(f"Engine {engine} not implemented")

    def build_dask(self, params: dict[str, Any]) -> DaskProgram:
        # Using node._parent_id_dict.items()
        # as kwargs in the task spec with apply would treat n.id as a literal string
        # and not as a task key.
        # We need to use (dict, [[k, v], ...]) instead, where the inner structure must
        # be a list of lists; tuples don't work.

        task_graph = {}
        for node, params in self._depth_first_with_params(params):
            task_graph[node._id] = (
                dask_apply,
                node._func,
                (),
                (
                    dict,
                    [
                        [k, v]
                        for k, v in chain(
                            params.items(),
                            node._args.items(),
                            node._parent_id_dict().items(),
                        )
                    ],
                ),
            )
        return DaskProgram(task_graph=task_graph, root_name=self._id)

    def _parent_id_dict(self) -> dict[str, str]:
        return {name: parent._id for name, parent in self._parents.items()}


def node(*, name: str | None = None, params: tuple[str, ...] = ()):
    def decorator(func: Callable) -> Callable:
        def node_constructor(**kwargs):
            return Node(name=name, func=func, func_kwargs=kwargs, params=params)

        node_constructor.__name__ = func.__name__
        node_constructor.__node_wrapped__ = func

        return node_constructor

    return decorator


@dataclass()
class Data:
    x: float
    monitor: Data | None = None


def divide(dividend: Data, divisor: Data) -> Data:
    return Data(x=dividend.x / divisor.x)


def get_monitor_attr(data: Data) -> Data | None:
    return data.monitor


@node(params=("filename",))
def load_nexus(filename: str) -> Data:
    return Data(x=len(filename), monitor=Data(x=len(filename) - 4))


@node(params=("fudge",))
def preprocess(
    data: Data,
    get_monitor: Callable[[Data], Data],
    normalize: Callable[[Data, Data], Data] = divide,
    fudge: float | None = None,
) -> Data:
    mon = get_monitor(data)
    if fudge is not None:
        data = Data(x=data.x * fudge)
    return normalize(data, mon)


@node()
def reduce(
    sample_data: Data, vana_data: Data, normalize: Callable[[Data, Data], Data]
) -> Data:
    return normalize(sample_data, vana_data)


def get_optional_args(argspec: inspect.FullArgSpec) -> set[str]:
    optional = (
        set(argspec.args[-len(argspec.defaults) :]) if argspec.defaults else set()
    )
    if argspec.kwonlydefaults:
        return optional | set(argspec.kwonlydefaults)
    return optional


def main() -> None:
    sample = preprocess(
        data=load_nexus(), get_monitor=get_monitor_attr
    )  # default normalize
    vana = sample.clone()
    workflow = reduce(sample_data=sample, vana_data=vana, normalize=divide)
    params = {
        "sample_data": {"data": {"filename": "sample_file.nxs"}},
        "vana_data": {"data": {"filename": "vana_file.nxs"}, "fudge": 0.9},
    }

    graph = workflow.build(params, engine="dask")
    # print(graph._task_graph)
    print("result:  ", graph.compute())

    s = len("sample_file.nxs") / (len("sample_file.nxs") - 4)
    v = len("vana_file.nxs") / (len("vana_file.nxs") - 4) * 0.9
    print("expected:", s / v)


if __name__ == "__main__":
    main()
