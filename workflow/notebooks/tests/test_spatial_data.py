import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_data import aggregate, weighted_output


class PreparationTests(unittest.TestCase):
    def test_weighted_delivery_uses_output_sign_and_hours(self):
        flows = pd.DataFrame({'alkaline': [-10.0, 3.0], 'PEM': [-20.0, -5.0]})
        result = weighted_output(flows, pd.Series([2.0, 4.0]))
        self.assertAlmostEqual(result['alkaline'], 20 / 1e6)
        self.assertAlmostEqual(result['PEM'], 60 / 1e6)

    def test_multiple_technologies_sum_to_same_region(self):
        values = pd.Series([2.0, 3.0], index=['alkaline', 'PEM'])
        locations = pd.Series(['BG1', 'BG1'], index=values.index)
        result = aggregate(values, locations, pd.Index(['BG1', 'BG2']))
        self.assertEqual(result.to_list(), [5.0, 0.0])

    def test_unknown_region_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            aggregate(pd.Series([2.0]), pd.Series(['unknown']), pd.Index(['BG1']))

    def test_infinite_potential_is_not_misrepresented_as_zero(self):
        with self.assertRaises(ValueError):
            aggregate(pd.Series([float('inf')]), pd.Series(['BG1']), pd.Index(['BG1']))

    def test_missing_snapshot_weight_fails(self):
        with self.assertRaises(ValueError):
            weighted_output(pd.DataFrame({'link': [-10, -20]}), pd.Series([1.0]))


if __name__ == '__main__':
    unittest.main()
