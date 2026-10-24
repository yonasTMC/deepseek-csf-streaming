from collections import OrderedDict


class SlotCache:
    """LRU slots; all experts needed by a call are protected from eviction."""

    def __init__(self, capacity, num_experts):
        if not 0 < capacity <= num_experts:
            raise ValueError("cache capacity must be within the expert inventory")
        self.capacity = capacity
        self.num_experts = num_experts
        self.resident = OrderedDict()
        self.free = list(reversed(range(capacity)))

    def reserve(self, experts):
        requested = list(dict.fromkeys(experts))
        if len(requested) > self.capacity:
            raise ValueError("routing chunk exceeds expert cache capacity")
        if any(not isinstance(e, int) or not 0 <= e < self.num_experts for e in requested):
            raise ValueError(f"invalid expert ID: {[e for e in requested if not isinstance(e, int) or not 0 <= e < self.num_experts][:8]}; inventory={self.num_experts}")
        protected = set(requested)
        misses = []
        for expert in requested:
            if expert in self.resident:
                self.resident.move_to_end(expert)
                continue
            if self.free:
                slot = self.free.pop()
            else:
                victim = next(e for e in self.resident if e not in protected)
                slot = self.resident.pop(victim)
            self.resident[expert] = slot
            misses.append((expert, slot))
        return {e: self.resident[e] for e in requested}, misses


def routing_chunks(rows, capacity):
    """Partition tokens without splitting or reordering any token's top-k."""
    start, selected = 0, set()
    for end, row in enumerate(rows):
        needed = set(row) - {-1}
        if len(needed) > capacity:
            raise ValueError("cache must hold every expert selected by one token")
        if len(selected | needed) > capacity:
            yield start, end
            start, selected = end, set()
        selected.update(needed)
    if rows:
        yield start, len(rows)
