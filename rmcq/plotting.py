"""Shared display limits for notebook accuracy plots."""
import math


def accuracy_ylim(values, *, padding=0.04, min_span=0.12, annotations=False):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return (0.0, 1.0)
    low, high = min(finite), max(finite)
    span = max(high - low + 2 * padding, min_span)
    middle = (low + high) / 2
    lower = max(0.0, min(middle - span / 2, 1.0 - span))
    upper = min(1.0, max(middle + span / 2, span))
    if annotations:
        upper = min(1.0, max(upper, high + 0.075))
    return lower, upper
