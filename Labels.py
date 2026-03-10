# Hi there! 🤗
# In this section, we integrate the ZigZag Indicator into our dataset.
# This will be imported into the "Preprocessing.py" file.
# Subsequently, we will train a DNN model using the ZigZag Indicator along with various other indicators found in "Indicators.py".
# Enjoy the process and have FUN!



from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional


class EndType(Enum):
    CLOSE = "close"
    HIGH_LOW = "high_low"


@dataclass
class ZigZagResult:
    point_type: Optional[str] = None


def _price(row, end_type: EndType, side: str) -> float:
    if end_type == EndType.CLOSE:
        if hasattr(row, "Close"):
            return float(getattr(row, "Close"))
        return float(getattr(row, "close"))

    if side == "high":
        if hasattr(row, "High"):
            return float(getattr(row, "High"))
        return float(getattr(row, "high"))

    if hasattr(row, "Low"):
        return float(getattr(row, "Low"))
    return float(getattr(row, "low"))


def get_zig_zag(quotes: Iterable, end_type: EndType = EndType.CLOSE,
                percent_change: float = 5) -> List[ZigZagResult]:
    rows = list(quotes)
    results = [ZigZagResult() for _ in rows]
    if len(rows) < 2:
        return results

    threshold = percent_change / 100.0
    trend = 0
    pivot_idx = 0
    pivot_price = _price(rows[0], end_type, "high")
    extreme_idx = 0
    extreme_price = pivot_price

    for i in range(1, len(rows)):
        high = _price(rows[i], end_type, "high")
        low = _price(rows[i], end_type, "low")

        if trend == 0:
            up_move = (high / pivot_price) - 1.0
            down_move = (pivot_price / low) - 1.0

            if up_move >= threshold and up_move >= down_move:
                results[pivot_idx].point_type = "L"
                trend = 1
                extreme_idx = i
                extreme_price = high
            elif down_move >= threshold and down_move > up_move:
                results[pivot_idx].point_type = "H"
                trend = -1
                extreme_idx = i
                extreme_price = low
            continue

        if trend == 1:
            if high >= extreme_price:
                extreme_price = high
                extreme_idx = i

            pullback = (extreme_price / low) - 1.0
            if pullback >= threshold:
                results[extreme_idx].point_type = "H"
                trend = -1
                pivot_idx = extreme_idx
                pivot_price = extreme_price
                extreme_idx = i
                extreme_price = low
            continue

        if low <= extreme_price:
            extreme_price = low
            extreme_idx = i

        rebound = (high / extreme_price) - 1.0
        if rebound >= threshold:
            results[extreme_idx].point_type = "L"
            trend = 1
            pivot_idx = extreme_idx
            pivot_price = extreme_price
            extreme_idx = i
            extreme_price = high

    return results
