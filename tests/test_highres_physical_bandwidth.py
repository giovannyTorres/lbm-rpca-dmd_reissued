"""Independent physical Fourier modes guard resolution and reference isolation."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python/scripts'))
import run_highres_fixed_bandwidth as experiment


class PhysicalBandwidthTests(unittest.TestCase):
    def fixture(self,n,length,poison_reference=False):
        coordinate=np.arange(n)*2*np.pi/n
        low=np.sin(4*coordinate)[:,None]*np.ones((1,n))
        high=.2*np.sin(16*coordinate)[:,None]*np.ones((1,n))
        coefficients=np.array([1.,1.2,.9])[:,None,None]
        observed=np.stack((.12+coefficients*(low+high),np.full((3,n,n),-.07)),axis=1)
        reference=np.full_like(observed,np.nan)if poison_reference else np.zeros_like(observed)
        geometry={'velocity':reference,'times':np.array([0.,.1,.2]),'dx':length/n,'dy':length/n,
                  'boundary':'periodic','mask':np.zeros((n,n),bool)}
        return observed,geometry,low,high,coefficients

    def test_cutoff_uses_physical_wavenumber_and_preserves_mean(self):
        for n in (64,128):
            for length in (2*np.pi,4*np.pi):
                with self.subTest(n=n,length=length):
                    observed,geometry,low,high,coefficients=self.fixture(n,length)
                    result,diagnostics=experiment.fixed_fit(observed,geometry,'identity')
                    # Integer mode 16 is physical k=16 on 2pi, k=8 on 4pi.
                    expected=np.stack((.12+coefficients*(low+(high if length>2*np.pi else 0)),
                                       np.full((3,n,n),-.07)),axis=1)
                    np.testing.assert_allclose(result,expected,atol=2e-14,rtol=0)
                    np.testing.assert_allclose(result.mean(axis=(-2,-1)),observed.mean(axis=(-2,-1)),atol=3e-15)
                    # FFT output and estimator views can sum along different
                    # strides; this diagnostic includes reduction roundoff.
                    self.assertLess(diagnostics['physical_bandwidth_prior']['mean_change_max'],128*np.finfo(float).eps)

    def test_estimation_never_reads_clean_velocity(self):
        observed,geometry,*_=self.fixture(64,2*np.pi,poison_reference=True)
        result,diagnostics=experiment.fixed_fit(observed,geometry,'identity')
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(diagnostics['physical_bandwidth_prior']['reference_velocity_used'])


if __name__=='__main__':unittest.main()
