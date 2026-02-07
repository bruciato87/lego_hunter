from __future__ import annotations

import unittest

from models import LegoHunterRepository


class ModelsRetryTests(unittest.TestCase):
    def test_with_retry_succeeds_after_transient_failures(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        repo.max_retries = 3
        repo.retry_base_delay = 0.01

        state = {"calls": 0}

        def flaky() -> str:
            state["calls"] += 1
            if state["calls"] < 3:
                raise RuntimeError("temporary")
            return "ok"

        result = repo._with_retry("flaky", flaky)

        self.assertEqual(result, "ok")
        self.assertEqual(state["calls"], 3)

    def test_with_retry_raises_after_max_attempts(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        repo.max_retries = 2
        repo.retry_base_delay = 0.01

        with self.assertRaises(RuntimeError):
            repo._with_retry("always_fail", lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_search_opportunities_empty_query_short_circuit(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        repo.max_retries = 1
        repo.retry_base_delay = 0.01

        self.assertEqual(repo.search_opportunities("   "), [])


if __name__ == "__main__":
    unittest.main()
