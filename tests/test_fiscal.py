from __future__ import annotations

import unittest

from fiscal import FiscalGuardian


class FakeRepo:
    def __init__(self, summary):
        self.summary = summary

    def get_fiscal_sales_summary(self, year=None):  # noqa: ANN001
        return self.summary


class FiscalGuardianTests(unittest.TestCase):
    def test_green_status_allows_sell(self) -> None:
        repo = FakeRepo({"_all": {"transactions": 10, "gross_total": 900.0}})
        guardian = FiscalGuardian(repo)

        status = guardian.check_safety_status()

        self.assertEqual(status["status"], "GREEN")
        self.assertTrue(status["allow_sell_signals"])
        self.assertTrue(status["allow_buy_signals"])

    def test_yellow_status_blocks_sell(self) -> None:
        repo = FakeRepo({"_all": {"transactions": 26, "gross_total": 1200.0}})
        guardian = FiscalGuardian(repo)

        status = guardian.check_safety_status()

        self.assertEqual(status["status"], "YELLOW")
        self.assertFalse(status["allow_sell_signals"])

    def test_red_status_blocks_sell(self) -> None:
        repo = FakeRepo({"_all": {"transactions": 29, "gross_total": 1500.0}})
        guardian = FiscalGuardian(repo)

        status = guardian.check_safety_status()

        self.assertEqual(status["status"], "RED")
        self.assertFalse(status["allow_sell_signals"])


if __name__ == "__main__":
    unittest.main()
