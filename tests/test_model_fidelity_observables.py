"""Independent analytic stresses and physical priors for the model comparison."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/scripts"))
import run_model_fidelity_benchmark as experiment


class GeometryOnly(dict):
    def __getitem__(self, key):
        if key not in ("dx", "dy"):
            raise AssertionError("The filter accessed more than geometry")
        return super().__getitem__(key)


class ModelFidelityObservablesTests(unittest.TestCase):
    def test_instantaneous_stress_uses_symmetric_tensor_frobenius(self):
        # Two independent incompressible analytic modes. Reference pure shear
        # has only xy stress; candidate Taylor-Green has xx and yy stresses.
        # Their inner product is zero, and the exact tensor norms are known.
        n = 32
        coordinate = 2*np.pi*np.arange(n)/n
        x, y = np.meshgrid(coordinate, coordinate)
        nu = .04
        clean = np.stack((np.sin(y), np.zeros_like(y)))[None]
        candidate = np.stack((np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y)))[None]
        clean_shear = np.abs(np.cos(y))[None]
        clean_stress = np.stack((np.zeros_like(y), nu*np.cos(y), np.zeros_like(y)))[None]
        metadata = dict(model="bingham", consistency=nu, flow_index=1.,
                        yield_stress=0., regularization_m=10., shear_epsilon=.001,
                        solvent_viscosity=0.)
        reference = dict(velocity=clean, dx=2*np.pi/n, dy=2*np.pi/n, metadata=metadata,
                         shear_rate=clean_shear, apparent_viscosity=np.full_like(clean_shear,nu),
                         deviatoric_stress=clean_stress)
        output = experiment.material_observables(reference, candidate)
        # Candidate dissipation is twice the reference dissipation for these
        # exact fields. Full tensor error sqrt(1 + 2) = sqrt(3).
        self.assertAlmostEqual(output["stress_frobenius_relative_l2"], np.sqrt(3), places=12)
        self.assertAlmostEqual(output["dissipation_relative_l2"], 1., places=12)
        self.assertAlmostEqual(output["apparent_viscosity_relative_l2"], 0., places=12)

    def test_newtonian_dissipation_has_known_amplitude_scaling(self):
        n = 32
        y = (2*np.pi*np.arange(n)/n)[:,None]*np.ones((1,n))
        clean = np.stack((np.sin(y), np.zeros_like(y)))[None]
        reference = dict(velocity=clean, dx=2*np.pi/n, dy=2*np.pi/n,
                         metadata=dict(model="newtonian", viscosity=.04))
        output = experiment.material_observables(reference, .5*clean)
        self.assertAlmostEqual(output["reference_dissipation"][0], .04*2*np.pi**2,places=12)
        self.assertAlmostEqual(output["dissipation_relative_l2"], .75, places=12)

    def test_polymer_memory_remains_unavailable_from_velocity(self):
        value=np.zeros((2,2,16,16))
        for model in ("oldroyd_b", "fene_p"):
            with self.subTest(model=model):
                result=experiment.material_observables(
                    dict(velocity=np.full_like(value,np.nan),dx=2*np.pi/16,
                         metadata=dict(model=model)),value)
                self.assertIsNone(result["stress_frobenius_relative_l2"])
                self.assertIsNone(result["dissipation_relative_l2"])

    def test_physical_prior_reads_only_geometry_and_preserves_dc(self):
        n=32
        y=(2*np.pi*np.arange(n)/n)[:,None]*np.ones((1,n))
        value=np.stack((.12+np.sin(4*y)+.2*np.sin(12*y),np.full_like(y,-.07)))[None]
        geometry=GeometryOnly(dx=2*np.pi/n,dy=2*np.pi/n)
        result,_=experiment.postprocess_field(value,geometry,"helmholtz_k10",10.)
        expected=np.stack((.12+np.sin(4*y),np.full_like(y,-.07)))[None]
        np.testing.assert_allclose(result,expected,atol=8e-15,rtol=0)
        np.testing.assert_allclose(result.mean(axis=(-2,-1)),value.mean(axis=(-2,-1)),atol=3e-16)


if __name__ == "__main__":
    unittest.main()
