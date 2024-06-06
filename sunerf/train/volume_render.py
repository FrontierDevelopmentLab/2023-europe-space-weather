from typing import Callable

import torch
from torch import nn


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod


def nerf_forward(rays_o: torch.Tensor,
                 rays_d: torch.Tensor,
                 times: torch.Tensor,
                 model: nn.Module,
                 sampler: Callable,
                 hierarchical_sampler: Callable,
                 rendering: Callable):
    r"""_summary_

        Compute forward pass through model.

        Args:
            rays_o (tensor): Origin of rays
            rays_d (tensor): Direction of rays
            times (tensor): Times of maps
            near (float): Beginning point of ray sampling (in solar radii, converted to pixels)
            far (float): Ending point of ray sampling (in solar radii, converted to pixels)
            encoding_fn (Callable[[torch.Tensor], torch.Tensor]): Encoding function for the inputs
            coarse_model (nn.Module): Function sampling coarse points along rays.
            sample_stratified: function that determines the points that will be sampled by the coarse model
            kwargs_sample_stratified (dict): Number of samples per ray, along with perturbation boolean.
            n_samples_hierarchical (int): Number of samples per ray.
            kwargs_sample_hierarchical: Contains the perturbation boolean for sampling (If set, applies noise to sample positions)
            fine_model (nn.Module): Function to refine the sampling along rays.
        Returns:
            outputs: Synthesized filtergrams/images.
        """
    # load weights for hierarchical sampling
    with torch.no_grad():
        # Sample query points along each ray.
        sample_out = sampler(rays_o, rays_d)
        query_points, z_vals = sample_out['points'], sample_out['z_vals']

        # add time to query points
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

        # Prepare points.
        flat_points = query_points_time.view(-1, 4)

        # Coarse model pass.
        raw = model(flat_points)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])
        log_rho = raw[..., 0]

        # Perform differentiable volume rendering to re-synthesize the filtergrams.
        out = rendering(log_rho, query_points, z_vals, rays_o, rays_d)
        weights = out['weights']

    # model pass.

    # Apply hierarchical sampling for fine query points.
    hierarchical_out = hierarchical_sampler(rays_o, rays_d, z_vals, weights)
    query_points, z_vals = hierarchical_out['points'], hierarchical_out['z_vals']

    # add time to query points = expand to dimensions of query points and slice one dimension
    exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
    query_points_time = torch.cat([query_points, exp_times], -1)

    # Prepare points
    flat_points = query_points_time.view(-1, 4)

    # Forward pass new samples through fine model.
    raw = model(flat_points)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])
    log_rho = raw[..., 0]

    # Perform differentiable volume rendering to re-synthesize the filtergrams.
    out_fine = rendering(log_rho, query_points, z_vals, rays_o, rays_d)

    # density and height maps
    density_map = out_fine['pixel_density']
    height_map_sun = out_fine['distance_from_sun']
    height_map_obs = out_fine['distance_from_obs']
    pixel_b = out_fine['pixel_B']

    return {'pixel_B': pixel_b,
            'density_map': density_map, 'distance_sun': height_map_sun, 'distance_obs': height_map_obs,
            'z_vals_stratified': sample_out['z_vals'], 'z_vals_hierarchical': hierarchical_out['z_vals'],
            }


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix
