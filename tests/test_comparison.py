"""Exercise visibility, alignment, and the comparison-only lip exclusion."""
import unittest
import numpy as np
import pandas as pd
from mouse_pose.comparison import visible_errors, comparison_summary
from mouse_pose.train import make_train_command


class ComparisonTests(unittest.TestCase):
    def data(self):
        keys = ['nose_tip', 'lowerlip', 'mouth', 'pupil_center_right']
        cols = pd.MultiIndex.from_product([['label'], keys, ['x', 'y', 'visible']])
        gt = pd.DataFrame([[0, 0, 2]*4, [0, 0, 1]*4], index=['a', 'b'], columns=cols)
        pred = gt.copy(); pred.loc['a', pd.IndexSlice[:, :, 'x']] = 3
        return gt, pred.iloc[::-1, ::-1]

    def test_visible_alignment(self):
        gt, pred = self.data(); errors = visible_errors(gt, pred)
        self.assertEqual(errors.loc['a', 'nose_tip'], 3)
        self.assertTrue(errors.loc['b'].isna().all())
        self.assertTrue(errors.pupil_center_right.isna().all())

    def test_lips_only_affect_summary(self):
        gt, pred = self.data(); errors = visible_errors(gt, pred)
        out = comparison_summary(errors, np.array([10, 10]))
        self.assertEqual(out['all']['points'], 3)
        self.assertEqual(out['no_lips']['points'], 2)
        self.assertAlmostEqual(out['no_lips']['normalized_mean'], .3)
        self.assertEqual(errors.loc['a', 'lowerlip'], 3)

    def test_reject_missing_frames(self):
        gt, pred = self.data()
        with self.assertRaises(ValueError): visible_errors(gt, pred.iloc[:1])

    def test_nonfinite_visible_rejected(self):
        gt, pred = self.data();pred.loc['a', ('label', 'nose_tip', 'x')] = np.nan
        with self.assertRaises(ValueError): visible_errors(gt, pred)

    def test_explicit_recipe(self):
        cmd = make_train_command('x.csv', 'vitb_dinov3', 1, 0, [], '/tmp/new',
                                 temperature='2', head_mode='shared', config_file='custom.yaml')
        self.assertEqual(cmd[:3], ['litpose', 'train', 'custom.yaml'])
        self.assertIn('model.backbone=vitb_dinov3', cmd)


if __name__ == '__main__': unittest.main()
