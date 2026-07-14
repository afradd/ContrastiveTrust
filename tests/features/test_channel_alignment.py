import unittest
import pandas as pd
import numpy as np

from src.features.channel_alignment import (
    TYPE_PATTERNS,
    classify_columns,
    build_typed_frame
)

class TestChannelAlignment(unittest.TestCase):
    def setUp(self):
        self.swat_sensors = [
            'LIT101.Pv', 'FIT101.Pv', 'FIT201.Pv', 'AIT201.Pv', 'AIT202.Pv', 'AIT203.Pv',
            'AIT301.Pv', 'AIT302.Pv', 'AIT303.Pv', 'LIT301.Pv', 'FIT301.Pv', 'DPIT301.Pv',
            'LIT401.Pv', 'FIT401.Pv', 'AIT401.Pv', 'AIT402.Pv', 'FIT501.Pv', 'FIT502.Pv',
            'FIT503.Pv', 'FIT504.Pv', 'AIT501.Pv', 'AIT502.Pv', 'AIT503.Pv', 'AIT504.Pv',
            'PIT501.Pv', 'PIT502.Pv', 'PIT503.Pv', 'FIT601.Pv'
        ]
        
        self.swat_actuators = [
            'MV101.Status', 'P101.Status', 'P102.Status', 'MV201.Status', 'P201.Status',
            'P202.Status', 'P203.Status', 'P204.Status', 'P205.Status', 'P206.Status',
            'P207.Status', 'P208.Status', 'MV301.Status', 'MV302.Status', 'MV303.Status',
            'MV304.Status', 'P301.Status', 'P302.Status', 'P401.Status', 'P402.Status',
            'P403.Status', 'P404.Status', 'UV401.Status', 'P501.Status', 'P502.Status',
            'MV501.Status', 'MV502.Status', 'MV503.Status', 'MV504.Status', 'P601.Status',
            'P602.Status', 'P603.Status'
        ]
        
        self.hai_features = [
            'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z',
            'P1_FT01', 'P1_FT01Z', 'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z',
            'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02D',
            'P1_PCV02Z', 'P1_PIT01', 'P1_PIT01_HH', 'P1_PIT02', 'P1_PP01AD', 'P1_PP01AR',
            'P1_PP01BD', 'P1_PP01BR', 'P1_PP02D', 'P1_PP02R', 'P1_PP04', 'P1_PP04D',
            'P1_PP04SP', 'P1_SOL01D', 'P1_SOL03D', 'P1_STSP', 'P1_TIT01', 'P1_TIT02',
            'P1_TIT03', 'P2_24Vdc', 'P2_ATSW_Lamp', 'P2_AutoGO', 'P2_AutoSD', 'P2_Emerg',
            'P2_MASW', 'P2_MASW_Lamp', 'P2_ManualGO', 'P2_ManualSD', 'P2_OnOff', 'P2_RTR',
            'P2_SCO', 'P2_SCST', 'P2_SIT01', 'P2_TripEx', 'P2_VIBTR01', 'P2_VIBTR02',
            'P2_VIBTR03', 'P2_VIBTR04', 'P2_VT01', 'P2_VTR01', 'P2_VTR02', 'P2_VTR03',
            'P2_VTR04', 'P3_FIT01', 'P3_LCP01D', 'P3_LCV01D', 'P3_LH01', 'P3_LIT01',
            'P3_LL01', 'P3_PIT01', 'P4_HT_FD', 'P4_HT_PO', 'P4_HT_PS', 'P4_LD',
            'P4_ST_FD', 'P4_ST_GOV', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01',
            'P4_ST_TT01', 'x1001_05_SETPOINT_OUT', 'x1001_15_ASSIGN_OUT',
            'x1002_07_SETPOINT_OUT', 'x1002_08_SETPOINT_OUT', 'x1003_10_SETPOINT_OUT',
            'x1003_18_SETPOINT_OUT', 'x1003_24_SUM_OUT'
        ]
        
        self.all_columns = self.swat_sensors + self.swat_actuators + self.hai_features

    def test_all_columns_classified_exactly_once(self):
        """Asserts that every real SWaT and HAI feature column is classified into exactly 
        one of the six types, with nothing left unmatched or multiply matched.
        """
        for col in self.all_columns:
            matched_types = []
            for name, pattern in TYPE_PATTERNS:
                if pattern.search(col):
                    matched_types.append(name)
            
            self.assertEqual(
                len(matched_types), 1,
                f"Column '{col}' must match exactly 1 category, but matched {len(matched_types)}: {matched_types}"
            )

    def test_classify_columns_structure(self):
        """Tests that classify_columns returns the correct dictionary structure."""
        groups = classify_columns(self.all_columns)
        
        # Verify 6 physics groups exist
        expected_keys = {"flow", "level", "pressure", "temperature", "analyzer", "actuator"}
        self.assertEqual(set(groups.keys()), expected_keys)
        
        # Verify total classified count equals total columns
        total_classified = sum(len(cols) for cols in groups.values())
        self.assertEqual(total_classified, len(self.all_columns))

    def test_build_typed_frame(self):
        """Tests the projection to the 6-channel physical variable space."""
        # Create a tiny mock dataframe
        df = pd.DataFrame({
            'LIT101.Pv': [100.0, 110.0, 120.0],  # level
            'FIT101.Pv': [1.0, 2.0, 3.0],        # flow
            'P1_TIT01': [20.0, 22.0, 24.0],      # temperature
            't_stamp': [1, 2, 3],                # keep
            'label': [0, 1, 0]                   # keep
        })
        
        feature_columns = ['LIT101.Pv', 'FIT101.Pv', 'P1_TIT01']
        typed_df = build_typed_frame(df, feature_columns, keep=['t_stamp', 'label'])
        
        # Should have 6 physics channels + 2 kept columns = 8 columns total
        self.assertEqual(len(typed_df.columns), 8)
        
        # Check standard physics channels exist
        for col in ["flow", "level", "pressure", "temperature", "analyzer", "actuator"]:
            self.assertIn(col, typed_df.columns)
            
        # Check zero-filling for absent categories
        np.testing.assert_array_equal(typed_df['pressure'].values, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(typed_df['analyzer'].values, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(typed_df['actuator'].values, [0.0, 0.0, 0.0])
        
        # Check normalization (mean 0, std 1 over time for single columns)
        # LIT101 has values 100, 110, 120 (mean=110, std=8.16)
        self.assertAlmostEqual(typed_df['level'].mean(), 0.0, places=5)
        self.assertAlmostEqual(typed_df['flow'].mean(), 0.0, places=5)
        self.assertAlmostEqual(typed_df['temperature'].mean(), 0.0, places=5)
        
        # Check kept columns
        np.testing.assert_array_equal(typed_df['t_stamp'].values, [1, 2, 3])
        np.testing.assert_array_equal(typed_df['label'].values, [0, 1, 0])

if __name__ == '__main__':
    unittest.main()
