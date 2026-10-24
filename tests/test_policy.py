import unittest
import random
from csf_streaming.policy import SlotCache, routing_chunks


class CacheTests(unittest.TestCase):
    def test_padding_does_not_consume_a_cache_slot(self):
        self.assertEqual(list(routing_chunks([[-1,-1], [0,1], [-1,1]], 2)), [(0,3)])

    def test_protected_hit_is_not_evicted_before_it_is_visited(self):
        c = SlotCache(2, 10)
        c.reserve([0, 1])
        mapping, misses = c.reserve([2, 0])
        self.assertEqual(mapping, {2: 1, 0: 0})
        self.assertEqual(misses, [(2, 1)])

    def test_repeated_routes_hit_and_lru_evicts(self):
        c = SlotCache(3, 10)
        c.reserve([0, 1, 2])
        self.assertEqual(c.reserve([0, 0])[1], [])
        self.assertEqual(c.reserve([3])[1], [(3, 1)])

    def test_invalid_request_does_not_mutate_cache(self):
        c = SlotCache(2, 4)
        c.reserve([1])
        before = list(c.resident.items())
        for ids in ([0, 1, 2], [-1], [4]):
            with self.assertRaises(ValueError):
                c.reserve(ids)
            self.assertEqual(list(c.resident.items()), before)

    def test_chunks_preserve_token_routes(self):
        rows = [[0, 1], [1, 2], [3, 4], [4, 5]]
        self.assertEqual(list(routing_chunks(rows, 3)), [(0, 2), (2, 4)])
        self.assertEqual(list(routing_chunks([], 3)), [])
        with self.assertRaises(ValueError):
            list(routing_chunks([[0, 1, 2]], 2))

    def test_random_routes_never_alias_live_experts(self):
        rng = random.Random(71)
        cache = SlotCache(16, 384)
        for _ in range(1000):
            requested = rng.sample(range(384), rng.randint(1, 16))
            mapping, misses = cache.reserve(requested)
            self.assertEqual(set(mapping), set(requested))
            self.assertEqual(len(set(mapping.values())), len(mapping))
            self.assertTrue(all(0 <= s < 16 for s in mapping.values()))
            self.assertEqual(len(cache.resident) + len(cache.free), 16)
            self.assertEqual(cache.reserve(requested)[1], [])


if __name__ == "__main__":
    unittest.main()
