from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set

from nnarith.config import DataSweep


@dataclass(frozen=True)
class EncodingSpec:
    base: int
    operand_digits: int
    result_digits: int

    @property
    def input_size(self) -> int:
        # Features: sign + digits for each operand plus single op-code slot
        return (2 * (1 + self.operand_digits)) + 1

    @property
    def target_size(self) -> int:
        # Targets: sign + digits for the result
        return 1 + self.result_digits


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


def encode_number(value: int, digits: int, base: int) -> List[float]:
    sign = 1.0 if value < 0 else 0.0
    magnitude = abs(value)
    encoded_digits = [0.0] * digits
    for position in range(digits - 1, -1, -1):
        encoded_digits[position] = (magnitude % base) / (base - 1)
        magnitude //= base
    return [sign] + encoded_digits


def decode_number(encoded: Sequence[float], base: int) -> int:
    if not encoded:
        raise ValueError("encoded sequence must not be empty")
    sign_bit = encoded[0] >= 0.5
    magnitude_digits = encoded[1:]
    magnitude = 0
    for digit_value in magnitude_digits:
        clamped = min(max(digit_value, 0.0), 1.0)
        digit = int(round(clamped * (base - 1)))
        magnitude = magnitude * base + digit
    return -magnitude if sign_bit else magnitude


__all__ = ["EncodingSpec", "compute_encoding", "decode_number", "encode_number", "required_digits"]
