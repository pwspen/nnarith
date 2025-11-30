from __future__ import annotations

from typing import Set

from gen import EncodingSpec

from nnarith.config import DataSweep


def required_digits(max_abs_value: int, base: int) -> int:
    digits = 1
    threshold = base
    while max_abs_value >= threshold:
        digits += 1
        threshold *= base
    return digits


def compute_encoding(sweep: DataSweep) -> EncodingSpec:
    candidates: Set[int] = {sweep.train.minimum, sweep.train.maximum}
    for split in sweep.evaluations.values():
        candidates.add(split.minimum)
        candidates.add(split.maximum)

    operand_max_abs = max(abs(value) for value in candidates)

    result_max_abs = 0
    for left in candidates:
        for right in candidates:
            for operation in sweep.operations:
                result_max_abs = max(result_max_abs, abs(operation(left, right)))

    operand_digits = required_digits(operand_max_abs, sweep.base)
    result_digits = required_digits(result_max_abs, sweep.base)
    return EncodingSpec(base=sweep.base, operand_digits=operand_digits, result_digits=result_digits)


__all__ = ["compute_encoding", "required_digits"]
