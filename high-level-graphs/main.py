from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator

from rich import print


def get_args(func: Callable) -> tuple[set[str], set[str]]:
    a = inspect.getfullargspec(func)
    optional = set(a.args[-len(a.defaults) :]) if a.defaults else set()
    return set(a.args) - optional, optional


class Node:
    def __init__(
        self, func: Callable, func_kwargs: dict[str, Any], params: tuple[str]
    ) -> None:
        self._func = func
        args = get_args(func)

        self._parents = {k: v for k, v in func_kwargs.items() if isinstance(v, Node)}

        kw_params = {k: v for k, v in func_kwargs.items() if not isinstance(v, Node)}
        self._param_names = set(params) | set(kw_params)
        self._params = kw_params
        self._optional_params = args[1]

    def set_param(self, name: str, value: Any) -> None:
        self._params[name] = value

    def compute(self):
        print("Computing", self._func.__name__)
        assert set(self._params.keys()) | self._optional_params == set(
            self._param_names
        ), "Not all params are set"
        return self._func(
            **self._params, **{k: p.compute() for k, p in self._parents.items()}
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

        return node_constructor

    return decorator


@dataclass
class Data:
    x: float
    monitor: Data | None = None


def divide(dividend: Data, divisor: Data) -> Data:
    return Data(x=dividend.x / divisor.x)


def get_monitor(data: Data) -> Data | None:
    return data.monitor


@node(params=("filename",))
def load_nexus(filename: str) -> Data:
    return Data(x=len(filename), monitor=Data(x=len(filename) - 4))


@node(params=("fudge",))
def preprocess(
    data: Data,
    get_monitor: Callable[[Data], Data],
    normalize: Callable[[Data, Data], Data],
    fudge: float | None = None,
) -> Data:
    mon = get_monitor(data)
    if fudge is not None:
        data = Data(x=data.x * fudge)
    return normalize(data, mon)


@node()
def reduce(sample_data: Data, vana_data: Data, normalize: Callable) -> Data:
    return normalize(sample_data, vana_data)


def main() -> None:
    sample = preprocess(data=load_nexus(), get_monitor=get_monitor, normalize=divide)
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
