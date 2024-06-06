import os
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from skimage.metrics import structural_similarity
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from sunerf.evaluation.callback import plot_samples
from sunerf.model.model import NeRF
from sunerf.train.rendering import ThompsonScattering
from sunerf.train.sampling import SphericalSampler, HierarchicalSampler
from sunerf.train.scaling import ImageLogScaling
from sunerf.train.volume_render import nerf_forward, jacobian
from sunerf.utilities.data_loader import NeRFDataModule


class SuNeRFModule(LightningModule):
    def __init__(self, Mm_per_ds, seconds_per_dt,
                 lambda_continuity=0, lambda_radial_regularization=0, lambda_velocity_regularization=0,
                 lambda_tB=1.0, lambda_pB=1.0,
                 sampling_config={'near': -1.4e5, 'far': 1.4e5},
                 model_config={}, lr_config={'start': 1e-4, 'end': 1e-5, 'iterations': 1e6},
                 hierachical_sampling_config={},
                 image_scaling_config={},
                 cmap=None):
        super().__init__()
        self.cmap = cmap if cmap is not None else 'viridis'

        self.lambda_continuity = lambda_continuity
        self.lambda_radial_regularization = lambda_radial_regularization
        self.lambda_velocity_regularization = lambda_velocity_regularization
        self.lambda_tB = lambda_tB
        self.lambda_pB = lambda_pB

        self.lr_config = lr_config

        # setup sampling strategies
        self.sample_stratified = SphericalSampler(**sampling_config, Mm_per_ds=Mm_per_ds)
        self.sample_hierarchical = HierarchicalSampler(**hierachical_sampling_config)
        self.image_scaling = ImageLogScaling(**image_scaling_config)
        self.rendering = ThompsonScattering(Mm_per_ds=Mm_per_ds)

        # Model Loading
        self.rho_model = NeRF(d_output=1)
        self.v_model = NeRF(d_output=3)
        self.mse_loss = nn.MSELoss()

        v_wind = 400e-3 / Mm_per_ds * seconds_per_dt  # Mm/s --> ds/dt
        self.register_buffer('v_wind', torch.tensor(v_wind, dtype=torch.float32))


    def configure_optimizers(self):
        params = list(self.rho_model.parameters()) + list(self.v_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr_config['start'])
        gamma = (self.lr_config['end'] / self.lr_config['start']) ** (1 / self.lr_config['iterations'])

        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)  # decay over N iterations

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        rays, time, target_img = batch['tracing']['rays'], batch['tracing']['times'], batch['tracing']['images']
        rays_o, rays_d = rays[:, 0], rays[:, 1]

        outputs = nerf_forward(rays_o, rays_d, time,
                               model=self.rho_model, sampler=self.sample_stratified,
                               hierarchical_sampler=self.sample_hierarchical,
                               rendering=self.rendering)

        # Check for any numerical issues.
        for k, v in outputs.items():
            assert not torch.isnan(v).any(), f"! [Numerical Alert] {k} contains NaN."
            assert not torch.isinf(v).any(), f"! [Numerical Alert] {k} contains Inf."

        # backpropagation
        pred_img = self.image_scaling(outputs['pixel_B'])
        target_img = self.image_scaling(target_img)
        b_loss = self.lambda_tB * self.mse_loss(pred_img[..., 0], target_img[..., 0]) + \
                 self.lambda_pB * self.mse_loss(pred_img[..., 1], target_img[..., 1])

        # PINN

        query_points = batch['random']
        query_points.requires_grad = True

        radial = query_points[..., :3] / (torch.norm(query_points[..., :3], dim=-1, keepdim=True) + 1e-8)
        solar_wind = self.v_wind * radial

        log_rho = self.rho_model(query_points)
        rho = 10 ** log_rho
        velocity = self.v_model(query_points)

        jac_rho = jacobian(log_rho, query_points)
        jac_v = jacobian(velocity, query_points)

        jac_matrix = torch.cat([jac_rho, jac_v], dim=1)

        dlogRho_dx = jac_matrix[:, 0, 0]
        dVx_dx = jac_matrix[:, 1, 0]
        dVy_dx = jac_matrix[:, 2, 0]
        dVz_dx = jac_matrix[:, 3, 0]

        dlogRho_dy = jac_matrix[:, 0, 1]
        dVx_dy = jac_matrix[:, 1, 1]
        dVy_dy = jac_matrix[:, 2, 1]
        dVz_dy = jac_matrix[:, 3, 1]

        dlogRho_dz = jac_matrix[:, 0, 2]
        dVx_dz = jac_matrix[:, 1, 2]
        dVy_dz = jac_matrix[:, 2, 2]
        dVz_dz = jac_matrix[:, 3, 2]

        dlogRho_dt = jac_matrix[:, 0, 3]
        dVx_dt = jac_matrix[:, 1, 3]
        dVy_dt = jac_matrix[:, 2, 3]
        dVz_dt = jac_matrix[:, 3, 3]

        grad_logRho = torch.stack([dlogRho_dx, dlogRho_dy, dlogRho_dz], -1)
        div_V = (dVx_dx + dVy_dy + dVz_dz)
        v_dot_grad_Rho = (velocity * grad_logRho).sum(-1)
        continuity_eq = dlogRho_dt + div_V + v_dot_grad_Rho
        continuity_loss = continuity_eq.pow(2).mean()

        # div_v = (dVx_dx + dVy_dy + dVz_dz)
        # grad_rho = torch.stack([dRho_dx, dRho_dy, dRho_dz], -1)
        # v_dot_grad_rho = (velocity * grad_rho).sum(-1)
        # continuity_loss = dRho_dt + rho * div_v + v_dot_grad_rho
        # continuity_loss = continuity_loss.abs()
        # continuity_loss = continuity_loss.mean() / rho.detach().mean()

        # regularize vectors to point radially outwards
        # v_unit = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        # radial = query_points[..., :3] / (torch.norm(query_points[..., :3], dim=-1, keepdim=True) + 1e-8)
        # radial_regularization_loss = v_unit - radial
        # radial_regularization_loss = (torch.norm(radial_regularization_loss, dim=-1) ** 2).mean()
        #
        # # regularize velocity
        velocity_norm = torch.norm(velocity, dim=-1)
        min_velocity_regularization_loss = torch.relu(self.v_wind - velocity_norm).pow(2).mean()

        velocity_unit = velocity / (torch.norm(velocity, dim=-1, keepdim=True) + 1e-8)
        radial_regularization_loss = (1 - torch.sum(velocity_unit * radial, dim=-1)).pow(2).mean()
        # velocity_regularization_loss = torch.clip(torch.norm(v - solar_wind, dim=-1) - self.v_tolerance, min=0).mean()

        loss = (b_loss +
                self.lambda_continuity * continuity_loss +
                self.lambda_velocity_regularization * min_velocity_regularization_loss +
                self.lambda_radial_regularization * radial_regularization_loss)

        with torch.no_grad():
            psnr = -10. * torch.log10(b_loss)

        # log results to WANDB
        self.log("loss", loss)
        self.log("train", {
            'brightness': b_loss,
            'continuity': continuity_loss,
            'velocity_reg': min_velocity_regularization_loss,
            'radial_reg': radial_regularization_loss,
            'total': loss,
            'psnr': psnr
        })

        return loss

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.scheduler.get_last_lr()[0] > 5e-5:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])

    def validation_step(self, batch, batch_nb):
        rays, time, target_img = batch['rays'], batch['times'], batch['images']
        rays_o, rays_d = rays[:, 0], rays[:, 1]

        outputs = nerf_forward(rays_o, rays_d, time,
                               model=self.rho_model, sampler=self.sample_stratified,
                               hierarchical_sampler=self.sample_hierarchical,
                               rendering=self.rendering)

        distance = rays_o.pow(2).sum(-1).pow(0.5).mean()
        return {'target_img': target_img,
                'channel_map': outputs['pixel_B'],
                'distance_sun': outputs['distance_sun'],
                'distance_obs': outputs['distance_obs'],
                'density_map': outputs['density_map'],
                'z_vals_stratified': outputs['z_vals_stratified'],
                'z_vals_hierarchical': outputs['z_vals_hierarchical'],
                'distance': distance}

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        target_img = torch.cat([o['target_img'] for o in outputs])
        channel_map = torch.cat([o['channel_map'] for o in outputs])
        distance_sun = torch.cat([o['distance_sun'] for o in outputs])
        distance_obs = torch.cat([o['distance_obs'] for o in outputs])
        density_map = torch.cat([o['density_map'] for o in outputs])
        z_vals_stratified = torch.cat([o['z_vals_stratified'] for o in outputs])
        z_vals_hierarchical = torch.cat([o['z_vals_hierarchical'] for o in outputs])

        wh = int(np.sqrt(target_img.shape[0]))
        target_img = target_img.view(wh, wh, target_img.shape[1]).cpu().numpy()
        channel_map = channel_map.view(wh, wh, channel_map.shape[1]).cpu().numpy()
        distance_sun = distance_sun.view(wh, wh, -1).cpu().numpy()
        distance_obs = distance_obs.view(wh, wh, -1).cpu().numpy()
        density_map = density_map.view(wh, wh, -1).cpu().numpy()
        z_vals_stratified = z_vals_stratified.view(wh, wh, -1).cpu().numpy()
        z_vals_hierarchical = z_vals_hierarchical.view(wh, wh, -1).cpu().numpy()
        distance = outputs[0]['distance'].cpu().numpy().mean()

        # TODO move plotting to separate plot callback, remove coarse map
        plot_samples(channel_map, channel_map, distance_sun, distance_obs, density_map, target_img,
                     z_vals_stratified,
                     z_vals_hierarchical, distance=distance, cmap=self.cmap)

        channel_map = (np.log(channel_map) + 8) / (-4 + 8)
        target_img = (np.log(target_img) + 8) / (-4 + 8)
        val_loss = np.nanmean((channel_map - target_img) ** 2)
        mask = ~np.isnan(target_img[..., 0])

        channel_map_copy = channel_map.copy()
        channel_map_copy[~mask, :] = 0
        val_ssim_tB = structural_similarity(np.nan_to_num(target_img[..., 0], nan=0), channel_map_copy[..., 0],
                                            data_range=1)
        val_ssim_pB = structural_similarity(np.nan_to_num(target_img[..., 1], nan=0), channel_map_copy[..., 1],
                                            data_range=1)
        val_ssim = np.mean([val_ssim_tB, val_ssim_pB])
        val_psnr = -10. * np.log10(val_loss)
        self.log("Validation Loss", val_loss)
        self.log("Validation PSNR", val_psnr)
        self.log("Validation SSIM", val_ssim)

        return {'progress_bar': {'val_loss': val_loss,
                                 'val_psnr': val_psnr,
                                 'val_ssim': val_ssim},
                'log': {'val/loss': val_loss, 'val/psnr': val_psnr, 'val/ssim': val_ssim}}


def save_state(sunerf: SuNeRFModule, data_module: NeRFDataModule, save_path, config_data):
    output_path = '/'.join(save_path.split('/')[0:-1])
    os.makedirs(output_path, exist_ok=True)
    torch.save({'model': {'rho': sunerf.rho_model, 'v': sunerf.v_model},
                'wavelength': data_module.wavelength,
                'sampling': {'hierarchical': sunerf.sample_hierarchical, 'stratified': sunerf.sample_stratified},
                'image_scaling': sunerf.image_scaling,
                'rendering': sunerf.rendering,
                'config': config_data,
                'times': data_module.times, 'radius_range': data_module.radius_range,
                'Mm_per_ds': data_module.Mm_per_ds, 'seconds_per_dt': data_module.seconds_per_dt,
                'test_kwargs': data_module.test_kwargs,
                'ref_time': data_module.ref_time,
                },
               save_path)
