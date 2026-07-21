from __future__ import annotations

import unittest

import pandas as pd

from run_real_data_tests import cluster_robust_slope, code_string


class AnalysisHelpersTest(unittest.TestCase):
    def test_code_string_preserves_old_and_new_aggregate_codes(self) -> None:
        values = pd.Series([99940.0, "9994000", "", None])
        result = code_string(values)
        self.assertEqual(result.iloc[0], "99940")
        self.assertEqual(result.iloc[1], "9994000")
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_fixed_effect_slope_recovers_known_relationship(self) -> None:
        rows = []
        for cluster in range(20):
            for year in (2000, 2001, 2002):
                x = cluster % 4 + year - 2000
                rows.append(
                    {
                        "y": 2.0 * x + 10.0 * (year - 2000),
                        "x": x,
                        "year": year,
                        "cluster": cluster,
                    }
                )
        result = cluster_robust_slope(
            pd.DataFrame(rows),
            outcome="y",
            regressor="x",
            fixed_effect="year",
            cluster="cluster",
        )
        self.assertAlmostEqual(result["estimate"], 2.0, places=10)
        self.assertEqual(result["observations"], 60)
        self.assertEqual(result["clusters"], 20)


if __name__ == "__main__":
    unittest.main()
