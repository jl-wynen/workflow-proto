from typing import Any

import scipp as sc
import scippneutron as scn


class DataHandle:
    ...


def load_sample_data(path: str) -> DataHandle:
    ...


def get_proton_charge(data: DataHandle) -> DataHandle:
    ...


def normalize_by_proton_charge(data: DataHandle,
                               charge: DataHandle) -> DataHandle:
    ...


def to_dspacing(data: DataHandle) -> DataHandle:
    ...


def associated_vanadium_path(data: DataHandle) -> str:  # str or Handle[str]?
    ...


def process(data: DataHandle) -> DataHandle:
    charge_normalized = normalize_by_proton_charge(
        data,
        charge=get_proton_charge(data))
    in_dspacing = to_dspacing(charge_normalized)
    return in_dspacing


def main() -> None:
    raw_sample_data = load_sample_data("<...>")
    sample_in_dspacing = process(raw_sample_data)

    raw_vana_data = load_sample_data(associated_vanadium_path(raw_sample_data))
    vana_in_dspacing = process(raw_vana_data)

    normalized = sample_in_dspacing / vana_in_dspacing
    normalized.plot()


if __name__ == "__main__":
    main()
