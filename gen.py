
from dataclasses import dataclass
from random import Random
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

Operation = Callable[[int, int], int]


@dataclass(frozen=True)
class ArithmeticDatasets:
    train: TensorDataset
    test: TensorDataset
    input_size: int
    target_size: int
    base: int
    operand_digits: int
    result_digits: int


@dataclass(frozen=True)
class EncodingSpec:
    base: int
    operand_digits: int
    result_digits: int

    @property
    def input_size(self) -> int:
        return (2 * (1 + self.operand_digits)) + 1

    @property
    def target_size(self) -> int:
        return 1 + self.result_digits


def generate_dataset_for_range(
    min_value: int,
    max_value: int,
    operations: Sequence[Operation],
    base: int,
    *,
    samples: Optional[int] = None,
    seed: Optional[int] = None,
    rng: Optional[Random] = None,
    encoding: Optional[EncodingSpec] = None,
) -> Tuple[TensorDataset, EncodingSpec, int, int]:
    if base < 2:
        raise ValueError("base must be >= 2")
    if not operations:
        raise ValueError("operations must not be empty")
    if min_value > max_value:
        raise ValueError("min value must be <= max value for the split")

    local_rng = rng
    if local_rng is None and seed is not None:
        local_rng = Random(seed)

    ops = list(operations)
    op_count = len(ops)
    operand_max_abs = max(abs(min_value), abs(max_value))

    examples = _collect_examples(
        min_value,
        max_value,
        ops,
        sample_count=samples,
        rng=local_rng,
    )

    result_max_abs = max(abs(example[3]) for example in examples) if examples else 0

    if encoding is None:
        operand_digits = _required_digits(operand_max_abs, base)
        result_digits = _required_digits(result_max_abs, base)
        encoding = EncodingSpec(base=base, operand_digits=operand_digits, result_digits=result_digits)
    else:
        if encoding.base != base:
            raise ValueError("encoding.base does not match requested base")
        required_operand_digits = _required_digits(operand_max_abs, base)
        if required_operand_digits > encoding.operand_digits:
            raise ValueError(
                "provided encoding.operand_digits is too small for the requested operand range"
            )
        required_result_digits = _required_digits(result_max_abs, base)
        if required_result_digits > encoding.result_digits:
            raise ValueError(
                "provided encoding.result_digits is too small for the generated results"
            )

    inputs, targets = _encode_examples(
        examples,
        encoding.operand_digits,
        encoding.result_digits,
        base,
        op_count,
    )
    dataset = TensorDataset(inputs, targets)
    return dataset, encoding, operand_max_abs, result_max_abs


def generate_arithmetic_datasets(
    train_min: int,
    train_max: int,
    test_min: int,
    test_max: int,
    operations: Sequence[Operation],
    base: int,
    *,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> ArithmeticDatasets:
    """
    Build train/test TensorDatasets encoding binary arithmetic problems.

    Features = sign(operand1) + base-N digits(operand1) +
                sign(operand2) + base-N digits(operand2) +
                single scalar op-code in [0, 1].

    Targets = sign(result) + base-N digits(result).
    The digit width is fixed by the largest operand/result magnitude across
both splits.
    """
    if base < 2:
        raise ValueError("base must be >= 2")
    if not operations:
        raise ValueError("operations must not be empty")
    if train_min > train_max or test_min > test_max:
        raise ValueError("min value must be <= max value for each split")

    base_rng = Random(seed) if seed is not None else None
    train_rng = (
        Random(base_rng.getrandbits(64)) if base_rng is not None else None
    )
    test_rng = (
        Random(base_rng.getrandbits(64)) if base_rng is not None else None
    )

    train_dataset, encoding, train_operand_max, train_result_max = generate_dataset_for_range(
        train_min,
        train_max,
        operations,
        base,
        samples=train_samples,
        rng=train_rng,
        encoding=None,
    )
    test_dataset, _, test_operand_max, test_result_max = generate_dataset_for_range(
        test_min,
        test_max,
        operations,
        base,
        samples=test_samples,
        rng=test_rng,
        encoding=encoding,
    )

    in_max = max(train_operand_max, test_operand_max)
    out_max = max(train_result_max, test_result_max)
    opcount = len(operations)

    print(f"\n\n{in_max=}, {out_max=}, {opcount=}")
    print(f"{base=}, {encoding.operand_digits=}, {encoding.result_digits=}")

    thing = ArithmeticDatasets(
        train=train_dataset,
        test=test_dataset,
        input_size=encoding.input_size,
        target_size=encoding.target_size,
        base=base,
        operand_digits=encoding.operand_digits,
        result_digits=encoding.result_digits,
    )
    print(f"input_size={thing.input_size}, target_size={thing.target_size}")
    print(f"operand_digits={thing.operand_digits}, result_digits={thing.result_digits}")
    return thing

def _collect_examples(
    min_value: int,
    max_value: int,
    operations: Sequence[Operation],
    sample_count: Optional[int] = None,
    rng: Optional[Random] = None,
) -> List[Tuple[int, int, int, int]]:
    op_count = len(operations)
    if op_count == 0:
        return []
    if sample_count is None:
        span = max_value - min_value + 1
        sample_count = span * span * op_count
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    random_source = rng or Random()
    examples: List[Tuple[int, int, int, int]] = []
    for _ in range(sample_count):
        left = random_source.randint(min_value, max_value)
        right = random_source.randint(min_value, max_value)
        op_index = random_source.randrange(op_count)
        result = operations[op_index](left, right)
        examples.append((left, right, op_index, int(result)))
    return examples


def _required_digits(max_abs_value: int, base: int) -> int:
    digits = 1
    threshold = base
    while max_abs_value >= threshold:
        digits += 1
        threshold *= base

    return digits


def _encode_examples(
    examples: Sequence[Tuple[int, int, int, int]],
    operand_digits: int,
    result_digits: int,
    base: int,
    op_count: int,
) -> Tuple[Tensor, Tensor]:
    inputs: List[List[float]] = []
    targets: List[List[float]] = []
    op_denominator = max(1, op_count - 1)

    for left, right, op_index, result in examples:
        feature = (
            encode_number(left, operand_digits, base)
            + encode_number(right, operand_digits, base)
            + [op_index / op_denominator]
        )
        target = encode_number(result, result_digits, base)
        inputs.append(feature)
        targets.append(target)

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def encode_number(value: int, digits: int, base: int) -> List[float]:
    sign = 1.0 if value < 0 else 0.0
    magnitude = abs(value)
    encoded_digits = [0.0] * digits
    for position in range(digits - 1, -1, -1):
        encoded_digits[position] = (magnitude % base) / (base - 1)
        magnitude //= base

    num = [sign] + encoded_digits
    # print(f"{value} -> {[round(n, 2) for n in num]}")
    return num


def decode_number(encoded: Sequence[float], base: int) -> int:
    """
    Decode a sign+digits representation back into an integer.
    """
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
