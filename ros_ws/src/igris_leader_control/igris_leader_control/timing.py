"""Low-overhead timing statistics for the leader serial loop."""

from collections import deque


def percentile(values, percent):
    """Return a linearly interpolated percentile, or ``None`` when empty."""
    if not values:
        return None
    if not 0.0 <= percent <= 100.0:
        raise ValueError('percent must be in the range [0, 100]')

    ordered = sorted(values)
    position = (len(ordered) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_ns(values):
    """Summarize nanosecond duration samples as milliseconds."""
    if not values:
        return None
    scale = 1.0e-6
    return {
        'count': len(values),
        'p50_ms': percentile(values, 50.0) * scale,
        'p95_ms': percentile(values, 95.0) * scale,
        'p99_ms': percentile(values, 99.0) * scale,
        'max_ms': max(values) * scale,
    }


class LeaderTimingWindow:
    """Collect bounded loop timing samples and produce periodic snapshots."""

    _METRIC_NAMES = ('loop', 'txrx', 'callback')

    def __init__(self, max_samples=2048):
        """Initialize a bounded sample window."""
        if max_samples < 1:
            raise ValueError('max_samples must be positive')
        self._samples = {
            name: deque(maxlen=max_samples) for name in self._METRIC_NAMES
        }
        self.attempts = 0
        self.published = 0
        self.dropped = 0

    def observe_ns(self, metric, duration_ns):
        """Record one non-negative duration for a known metric."""
        if metric not in self._samples:
            raise KeyError(metric)
        if duration_ns < 0:
            raise ValueError('duration_ns must not be negative')
        self._samples[metric].append(int(duration_ns))

    def record_attempt(self, published):
        """Record whether one serial read produced a published sample."""
        self.attempts += 1
        if published:
            self.published += 1
        else:
            self.dropped += 1

    def snapshot(self, reset=False):
        """Return counters and percentile summaries for the current window."""
        result = {
            'attempts': self.attempts,
            'published': self.published,
            'dropped': self.dropped,
            'metrics': {
                name: summarize_ns(tuple(samples))
                for name, samples in self._samples.items()
            },
        }
        if reset:
            self.reset()
        return result

    def reset(self):
        """Clear samples and counters for the next reporting window."""
        for samples in self._samples.values():
            samples.clear()
        self.attempts = 0
        self.published = 0
        self.dropped = 0
