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


def _initialize_zigzag_state(rows: List, end_type: EndType, percent_change: float):
    threshold = percent_change / 100.0
    trend = 0
    pivot_idx = 0
    pivot_price = _price(rows[0], end_type, "high")
    extreme_idx = 0
    extreme_price = pivot_price
    return threshold, trend, pivot_idx, pivot_price, extreme_idx, extreme_price


def _try_start_trend(high: float, low: float, threshold: float,
                     trend: int, pivot_idx: int, pivot_price: float,
                     extreme_idx: int, extreme_price: float, current_idx: int):
    up_move = (high / pivot_price) - 1.0
    down_move = (pivot_price / low) - 1.0

    point_type = None
    if up_move >= threshold and up_move >= down_move:
        point_type = "L"
        trend = 1
        extreme_idx = current_idx
        extreme_price = high
    elif down_move >= threshold and down_move > up_move:
        point_type = "H"
        trend = -1
        extreme_idx = current_idx
        extreme_price = low

    return point_type, trend, pivot_idx, pivot_price, extreme_idx, extreme_price


def _update_uptrend(high: float, low: float, threshold: float,
                    trend: int, pivot_idx: int, pivot_price: float,
                    extreme_idx: int, extreme_price: float, current_idx: int):
    point_type = None
    point_idx = None

    if high >= extreme_price:
        extreme_price = high
        extreme_idx = current_idx

    pullback = (extreme_price / low) - 1.0
    if pullback >= threshold:
        point_type = "H"
        point_idx = extreme_idx
        trend = -1
        pivot_idx = extreme_idx
        pivot_price = extreme_price
        extreme_idx = current_idx
        extreme_price = low

    return point_type, point_idx, trend, pivot_idx, pivot_price, extreme_idx, extreme_price


def _update_downtrend(high: float, low: float, threshold: float,
                      trend: int, pivot_idx: int, pivot_price: float,
                      extreme_idx: int, extreme_price: float, current_idx: int):
    point_type = None
    point_idx = None

    if low <= extreme_price:
        extreme_price = low
        extreme_idx = current_idx

    rebound = (high / extreme_price) - 1.0
    if rebound >= threshold:
        point_type = "L"
        point_idx = extreme_idx
        trend = 1
        pivot_idx = extreme_idx
        pivot_price = extreme_price
        extreme_idx = current_idx
        extreme_price = high

    return point_type, point_idx, trend, pivot_idx, pivot_price, extreme_idx, extreme_price


def _process_zigzag_row(results: List[ZigZagResult], high: float, low: float, threshold: float,
                        trend: int, pivot_idx: int, pivot_price: float,
                        extreme_idx: int, extreme_price: float, current_idx: int):
    if trend == 0:
        point_type, trend, pivot_idx, pivot_price, extreme_idx, extreme_price = _try_start_trend(
            high,
            low,
            threshold,
            trend,
            pivot_idx,
            pivot_price,
            extreme_idx,
            extreme_price,
            current_idx,
        )
        if point_type is not None:
            results[pivot_idx].point_type = point_type
        return trend, pivot_idx, pivot_price, extreme_idx, extreme_price

    if trend == 1:
        point_type, point_idx, trend, pivot_idx, pivot_price, extreme_idx, extreme_price = _update_uptrend(
            high,
            low,
            threshold,
            trend,
            pivot_idx,
            pivot_price,
            extreme_idx,
            extreme_price,
            current_idx,
        )
        if (point_type is not None) and (point_idx is not None):
            results[point_idx].point_type = point_type
        return trend, pivot_idx, pivot_price, extreme_idx, extreme_price

    point_type, point_idx, trend, pivot_idx, pivot_price, extreme_idx, extreme_price = _update_downtrend(
        high,
        low,
        threshold,
        trend,
        pivot_idx,
        pivot_price,
        extreme_idx,
        extreme_price,
        current_idx,
    )
    if (point_type is not None) and (point_idx is not None):
        results[point_idx].point_type = point_type
    return trend, pivot_idx, pivot_price, extreme_idx, extreme_price


def get_zig_zag(quotes: Iterable, end_type: EndType = EndType.CLOSE,
                percent_change: float = 5) -> List[ZigZagResult]:
    rows = list(quotes)
    results = [ZigZagResult() for _ in rows]
    if len(rows) < 2:
        return results

    threshold, trend, pivot_idx, pivot_price, extreme_idx, extreme_price = _initialize_zigzag_state(
        rows,
        end_type,
        percent_change,
    )

    for i in range(1, len(rows)):
        high = _price(rows[i], end_type, "high")
        low = _price(rows[i], end_type, "low")
        trend, pivot_idx, pivot_price, extreme_idx, extreme_price = _process_zigzag_row(
            results,
            high,
            low,
            threshold,
            trend,
            pivot_idx,
            pivot_price,
            extreme_idx,
            extreme_price,
            i,
        )

    return results
