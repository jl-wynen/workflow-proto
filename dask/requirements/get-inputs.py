"""
Extract input parameters from a task graph.

The inputs are stored in a flat dict, so it is not possible to determine
what function call they relate to (e.g. this example uses 2 calls to add).
Need to use the actual graph for this.
"""

import dask


@dask.delayed
def add(a, b):
    return a + b


@dask.delayed
def mul(a, b):
    return a * b


def find_pos_arg_inputs(graph, func_name, args):
    inputs = {}
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            inputs[f'{func_name}.{i}'] = arg
        elif arg not in graph:
            inputs[f'{func_name}.{i}'] = arg
    return inputs


def find_kwarg_inputs(graph, func_name, kwargs):
    inputs = {}
    for name, arg in kwargs.items():
        if not isinstance(arg, str):
            inputs[f'{func_name}.{name}'] = arg
        elif arg not in graph:
            inputs[f'{func_name}.{name}'] = arg
    return inputs


def find_inputs(graph: dict):
    inputs = {}
    for k, v in graph.items():
        if not isinstance(v, tuple):
            raise TypeError(f"Cannot handle graph spec with {v}")

        if v[0] == dask.utils.apply:
            func_name = v[1].__name__
            args = v[2]
            kwargs = dict(v[3][1])
            inputs.update(find_pos_arg_inputs(graph, func_name, args))
            inputs.update(find_kwarg_inputs(graph, func_name, kwargs))
        else:
            inputs.update(find_pos_arg_inputs(graph, v[0].__name__, v[1:]))
    return inputs


def main() -> None:
    a = 1
    b = 2
    c = ".c."
    n = add(b=a, a=add(b, 3))
    s = mul(c, n)

    print(s)
    print(s.compute())

    print(s.dask)
    print(s.dask.to_dict())
    print(find_inputs(s.dask.to_dict()))


if __name__ == '__main__':
    main()
