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

    def test_search_opportunities_merges_fields_without_or_filter_parser_issues(self) -> None:
        repo = object.__new__(LegoHunterRepository)
        repo.max_retries = 1
        repo.retry_base_delay = 0.01
        calls: list[tuple[str, str, int]] = []

        def fake_search(field: str, pattern: str, limit: int):  # noqa: ANN001
            calls.append((field, pattern, limit))
            if field == "set_name":
                return [
                    {
                        "set_id": "76441",
                        "source": "lego_proxy_reader",
                        "ai_investment_score": 80,
                        "market_demand_score": 90,
                        "last_seen_at": "2026-02-11T10:00:00Z",
                    }
                ]
            if field == "set_id":
                return [
                    {
                        "set_id": "76441",
                        "source": "lego_proxy_reader",
                        "ai_investment_score": 85,
                        "market_demand_score": 91,
                        "last_seen_at": "2026-02-11T11:00:00Z",
                    }
                ]
            return [
                {
                    "set_id": "10332",
                    "source": "amazon_proxy_reader",
                    "ai_investment_score": 70,
                    "market_demand_score": 80,
                    "last_seen_at": "2026-02-11T09:00:00Z",
                }
            ]

        repo._search_opportunities_by_field = fake_search  # type: ignore[attr-defined]

        rows = repo.search_opportunities("76441, Hogwarts (club)", limit=10)
        self.assertEqual([field for field, _, _ in calls], ["set_name", "set_id", "theme"])
        self.assertEqual(calls[0][1], "%76441, Hogwarts (club)%")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["set_id"], "76441")
        self.assertEqual(rows[0]["ai_investment_score"], 85)

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
            "shipping_in_cost": 9.0,
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
        self.assertEqual(calls["upserted"].shipping_in_cost, 6.0)


if __name__ == "__main__":
    unittest.main()
