from django.test import TestCase
from .backtests import backtest_strategy

# Create your tests here.
class BackTestCase(TestCase):
    def test_backtest_strategy(self):
        result = backtest_strategy('AAPL', 10000, 50, 200)
        self.assertIn('Final Monetary Value ($)', result)
        self.assertIn('Total Return', result)
        self.assertIn('Max Drawdown', result)
        self.assertIn('Number of Trades', result)
        self.assertGreaterEqual(result['Total Return'], 0)