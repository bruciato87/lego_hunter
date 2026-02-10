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

    def test_get_recent_ai_insights_empty_set_ids_short_circuit(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        repo.max_retries = 1
        repo.retry_base_delay = 0.01

        self.assertEqual(repo.get_recent_ai_insights([]), {})

    def test_register_portfolio_sale_full_removes_item_and_logs_fiscal(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        calls = {"deleted": None, "upserted": None, "fiscal": None}

        repo.get_portfolio_item = lambda set_id: {
            "set_id": "76441",
            "set_name": "Club dei Duellanti",
            "status": "holding",
            "quantity": 2,
            "purchase_price": 30.0,
            "purchase_date": "2026-01-01",
            "shipping_in_cost": 0.0,
        }
        repo.insert_fiscal_log = lambda record: calls.__setitem__("fiscal", record) or {"id": "f1"}
        repo.delete_portfolio_item = lambda set_id: calls.__setitem__("deleted", set_id)
        repo.upsert_portfolio_item = lambda record: calls.__setitem__("upserted", record)

        result = repo.register_portfolio_sale(
            set_id="76441",
            sale_price=89.9,
            quantity=2,
            platform="ebay",
        )

        self.assertTrue(result["sold_all"])
        self.assertEqual(result["remaining_quantity"], 0)
        self.assertEqual(calls["deleted"], "76441")
        self.assertIsNotNone(calls["fiscal"])
        self.assertIsNone(calls["upserted"])
        self.assertEqual(result["gross_amount"], 179.8)

    def test_register_portfolio_sale_partial_keeps_holding(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        calls = {"deleted": None, "upserted": None}

        repo.get_portfolio_item = lambda set_id: {
            "set_id": "76441",
            "set_name": "Club dei Duellanti",
            "status": "holding",
            "quantity": 3,
            "purchase_price": 30.0,
            "purchase_date": "2026-01-01",
            "shipping_in_cost": 0.0,
        }
        repo.insert_fiscal_log = lambda record: {"id": "f2"}
        repo.delete_portfolio_item = lambda set_id: calls.__setitem__("deleted", set_id)
        repo.upsert_portfolio_item = lambda record: calls.__setitem__("upserted", record) or {
            "set_id": record.set_id,
            "quantity": record.quantity,
            "status": record.status,
        }

        result = repo.register_portfolio_sale(
            set_id="76441",
            sale_price=89.9,
            quantity=1,
        )

        self.assertFalse(result["sold_all"])
        self.assertEqual(result["remaining_quantity"], 2)
        self.assertIsNone(calls["deleted"])
        self.assertIsNotNone(calls["upserted"])
        self.assertEqual(calls["upserted"].quantity, 2)
        self.assertEqual(calls["upserted"].status, "holding")


if __name__ == "__main__":
    unittest.main()
