import unittest
import numpy as np
from fluid_denoise.research.dns import taylor_green_reference
from fluid_denoise.research.physics import helmholtz_project, apply_postprocessing, vorticity_residual_metrics
from fluid_denoise.research.metrics import physical_metrics


class PhysicalProjectionTests(unittest.TestCase):
    def test_project_compressive_artifact_preserves_vorticity_and_mean(self):
        dataset = taylor_green_reference(nx=24, times=np.linspace(0,1,21))
        x,y=np.meshgrid(np.arange(24)*dataset['dx'],np.arange(24)*dataset['dy'])
        artifact=np.stack((3*np.cos(3*x)*np.cos(2*y),-2*np.sin(3*x)*np.sin(2*y)))
        candidate=dataset['velocity']+.2*artifact+np.array([.3,-.2])[None,:,None,None]
        projected,diagnostics=helmholtz_project(candidate,dataset['dx'],dataset['dy'])
        self.assertLess(diagnostics['mean_velocity_change_max'],1e-14)
        metrics=physical_metrics(dataset,projected)
        self.assertLess(metrics['extra_divergence_rms'],1e-13)
        self.assertLess(metrics['vorticity_relative_l2'],1e-13)
        self.assertGreater(metrics['velocity_relative_l2'],.1) # projection does not recover wrong mean
        again,_=helmholtz_project(projected,dataset['dx'],dataset['dy'])
        np.testing.assert_allclose(projected,again,atol=1e-14)
        self.assertLessEqual(np.linalg.norm(projected),np.linalg.norm(candidate))

    def test_even_grid_real_noise_satisfies_compatible_divergence(self):
        dataset=taylor_green_reference(nx=24,times=np.linspace(0,1,5))
        candidate=dataset['velocity']+np.random.default_rng(8).normal(size=dataset['velocity'].shape)
        projected,_=apply_postprocessing(dataset,candidate,'helmholtz')
        self.assertLess(physical_metrics(dataset,projected)['divergence_rms'],1e-13)

    def test_channel_projection_rejected(self):
        dataset=taylor_green_reference(nx=8,times=np.linspace(0,1,5));dataset['boundary']='channel'
        with self.assertRaises(ValueError): apply_postprocessing(dataset,dataset['velocity'],'helmholtz')

    def test_exact_taylor_green_temporal_residual_converges(self):
        residuals=[]
        for count in (21,41,81):
            dataset=taylor_green_reference(nx=24,times=np.linspace(0,1,count),viscosity=.2)
            metrics=vorticity_residual_metrics(dataset,dataset['velocity'])
            self.assertEqual(metrics['extra_vorticity_equation_residual_rms'],0.)
            residuals.append(metrics['reference_vorticity_equation_residual_rms'])
        self.assertGreater(residuals[0],3*residuals[1]); self.assertGreater(residuals[1],3*residuals[2])

    def test_velocity_only_viscoelastic_residual_not_mislabeled(self):
        dataset=taylor_green_reference(nx=8,times=np.linspace(0,1,5));dataset['metadata']['model']='oldroyd_b'
        metrics=vorticity_residual_metrics(dataset,dataset['velocity'])
        self.assertFalse(metrics['vorticity_equation_applicable'])
        self.assertIsNone(metrics['vorticity_equation_residual_rms'])

    def test_nyquist_second_derivative_not_erased(self):
        dataset=taylor_green_reference(nx=8,times=np.linspace(0,1,5),viscosity=1.)
        x,y=np.meshgrid(np.arange(8)*dataset['dx'],np.arange(8)*dataset['dy'])
        dataset['velocity'][:,0]=np.cos(y)*np.cos(4*x)
        dataset['velocity'][:,1]=0
        result=vorticity_residual_metrics(dataset,dataset['velocity'])
        self.assertAlmostEqual(result['vorticity_equation_residual_rms'],17/np.sqrt(2),places=11)

    def test_optional_empty_mask_geometry(self):
        dataset=taylor_green_reference(nx=8,times=np.linspace(0,1,5));dataset.pop('mask')
        projected,_=apply_postprocessing(dataset,dataset['velocity'],'helmholtz')
        self.assertTrue(vorticity_residual_metrics(dataset,projected)['vorticity_equation_applicable'])
