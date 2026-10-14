import unittest
import numpy as np
from fluid_denoise.research.noise import add_noise

class ExtendedNoiseTests(unittest.TestCase):
    def test_new_noise_models_same_realized_rms_and_determinism(self):
        clean=np.linspace(-1,1,12*2*8*8).reshape(12,2,8,8)
        mask=np.zeros((8,8),dtype=bool);mask[3:5,3:5]=True
        for kind in ('student_t','heteroscedastic','sensor_drift'):
            with self.subTest(kind=kind):
                noisy,meta=add_noise(clean,kind,.2,7,obstacle_mask=mask)
                again,_=add_noise(clean,kind,.2,7,obstacle_mask=mask)
                np.testing.assert_array_equal(noisy,again)
                np.testing.assert_array_equal(noisy[:,:,mask],clean[:,:,mask])
                self.assertAlmostEqual(meta['realized_relative_rms'],.2)

    def test_heteroscedastic_error_grows_with_local_speed(self):
        clean=np.ones((30,2,16,16));clean[:,:,:,:8]=.1
        noisy,_=add_noise(clean,'heteroscedastic',.2,17,noise_floor_fraction=0)
        self.assertGreater(np.std((noisy-clean)[:,:,:,8:]),5*np.std((noisy-clean)[:,:,:,:8]))

    def test_student_t_rejects_infinite_variance_specification(self):
        with self.assertRaises(ValueError): add_noise(np.ones((10,2,8,8)),'student_t',.2,1,degrees_of_freedom=2)
