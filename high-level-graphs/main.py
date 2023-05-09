from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator

from rich import print

# TODO macros: callables need to be expanded early to produce full graph
#  cannot merge those args into params

"""
    @node(params=("p1", "p2"))
    def foo(a, p1, b="x", p2="y"): ...

    foo(a=a_node)
    foo.set_params({"p1": "p"})

Arg types:
 - a: parent
 - b: optional given arg
 - p1: parameter
 - p2: optional parameter
 
Parents and given args must be specified in call to foo.
(Optional) parameters must be specified in set_params.
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


class Node:
    def __init__(
        self, func: Callable, func_kwargs: dict[str, Any], params: tuple[str]
    ) -> None:
        self._func = func
        args = classify_args(
            func, given_args=func_kwargs, designated_params=set(params)
        )
        self._parents = args.parents
        self._args = args.args
        self._param_names = args.param_names
        self._params = args.params
        self._optional = args.optional

    def set_param(self, name: str, value: Any) -> None:
        self._params[name] = value

    def compute(self):
        print("Computing", self._func.__name__)
        assert not set(
            self._param_names - self._params.keys() - self._optional
        ), "Not all params are set"
        return self._func(
            **self._params,
            **self._args,
            **{k: p.compute() for k, p in self._parents.items()},
        )

    @property
    def parents(self) -> Generator[tuple[str, Node], None, None]:
        yield from self._parents.items()

    def build(self, params: dict[str, Any]) -> None:
        for name, param in params.items():
            if name in self._parents:
                self._parents[name].build(param)
            else:
                self._params[name] = param


def node(params: tuple[str, ...] = ()):
    def decorator(func: Callable) -> Callable:
        def node_constructor(**kwargs):
            return Node(func=func, func_kwargs=kwargs, params=params)

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
    vana = deepcopy(sample)
    workflow = reduce(sample_data=sample, vana_data=vana, normalize=divide)
    params = {
        "sample_data": {"data": {"filename": "sample_file.nxs"}},
        "vana_data": {"data": {"filename": "vana_file.nxs"}, "fudge": 0.9},
    }
    workflow.build(params)
    print(workflow.compute())

    s = len("sample_file.nxs") / (len("sample_file.nxs") - 4)
    v = len("vana_file.nxs") / (len("vana_file.nxs") - 4) * 0.9
    print("expected:", s / v)


if __name__ == "__main__":
    main()
