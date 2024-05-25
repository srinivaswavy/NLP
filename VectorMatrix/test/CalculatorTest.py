import unittest

from Sample import Calculator


class CalculatorTestCase(unittest.TestCase):
    calc = Calculator()

    def test_addition(self):
        self.assertEqual(5, self.calc.add(2, 3))

    def test_subtraction(self):
        self.assertEqual(-1, self.calc.subtract(2, 3))

    def test_multiply(self):
        self.assertEqual(6, self.calc.multiply(2, 3))

    def test_divide(self):
        self.assertEqual(0.667, round(self.calc.divide(2, 3), 3))


if __name__ == '__main__':
    unittest.main()
