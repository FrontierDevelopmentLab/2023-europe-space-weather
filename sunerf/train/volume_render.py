import torch
from torch import nn
from typing import Tuple, Callable
import astropy.units as u
import numpy as np
from sunpy.io.special import read_genx
from scipy import interpolate
from astropy.visualization import ImageNormalize, AsinhStretch
from sunerf.train.sampling import sample_hierarchical
from sunerf.train.sampling import sample_non_uniform_box


def raw2outputs(raw: torch.Tensor, # (batch, sampling_points, density_e)
				query_points: torch.Tensor, # (batch, sampling_points, coord(x,y,z) )
				z_vals: torch.Tensor, # (batch, sampling_points, distance)
				rays_o: torch.Tensor,
				rays_d: torch.Tensor,
				raw_noise_std: float = 0.0
				) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	r"""
	Convert the raw NeRF output into electron density (1 model output).

	raw: output of NeRF, 2 values per sampled point
	z_vals: distance along the ray as measure from the origin
	"""

	# Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
	# compute line element (dz) for integration
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists[..., :1], dists], dim=-1)

	# Multiply each distance by the norm of its corresponding direction ray
	# to convert to real world distance (accounts for non-unit directions).
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	# Add noise to model's predictions for density. Can be used to
	# regularize network during training (prevents floater artifacts).
	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., -1].shape) * raw_noise_std

	# here we do stuff

	# For total and polarised brightness need... (Howard and Tappin 2009):
	# * Omega (half angular width of Sun - depends on distance to sun)
	# * z (distance from Q to observer)
	# * chi (scattering angle between observer and S)
	# * u (limb darkening coeff based on wavelength - constant for white light?)
	# * I0 (intensity/ of the Sun - will vary with solar cycle - look up table?)
	# * sigma_e (scattering constant - eqn 3)

	electron_density = raw[:, :, 0]

    # HOWARD AND TAPPIN 2009 FIG 3
	# working with units of solar radii
    # half angular width of Sun (angle between SQ and ST)
	s_q = query_points.pow(2).sum(-1).pow(0.5)
	s_t = 1
	omega = torch.asin(s_t / s_q)
	
	# z = distance Q to observer
	z = rays_o.pow(2).sum(-1).pow(0.5)

    # chi = scattering angle between line of sight and SQ (dot product)
	chi = torch.acos((rays_o * query_points).sum(-1) / (rays_o.pow(2).sum(-1).pow(0.5) * query_points.pow(2).sum(-1).pow(0.5) + 1e-6))
	
	# u = limb darkening coeff 
	# TODO hard coding for now [Ramos 2023]
	u = 0.63

	# sigma_e = thomson cross section [Howard and Tappin 2009 eqn 3]
	sigma_e = 7.95e-30 # m2/sr

	# I0 = intensity of the source (Sun) as a power per unit area (of the photosphere) per unit solid angle
	#    = mean solar radiance ( = irradiance / 4pi)
	# TODO average for now (move to lookup table later)
	I0 = 1361 / 4*torch.pi # W·sr−1·m−2 # (Prša et al., 2016)

	ln = torch.log((1 + torch.sin(omega)) / (torch.cos(omega) + 1e-6))
	cos2_sin = (torch.cos(omega) ** 2 / (torch.sin(omega) + 1e-6))
	A = torch.cos(omega) * torch.sin(omega) ** 2
	B = - (1/8) * (1 - 3 * torch.sin(omega) ** 2 - cos2_sin * (1 + 3 * torch.sin(omega) ** 2) * ln)
	C = (4 / 3) - torch.cos(omega) - torch.cos(omega) ** 3 / 3
	D = (1 / 8) * (5 + torch.sin(omega) ** 2 - cos2_sin * (5 - torch.sin(omega) ** 2) * ln)

    # equations 23, 24, 29
	intensity_T = I0 * torch.pi * sigma_e / (2 * z ** 2) * ((1 - u) * C + u * D)
	intensity_pB = I0 * torch.pi * sigma_e / (2 * z ** 2) * torch.sin(chi) ** 2 * ((1 - u) * A + u * B)

	intensity_tB = 2 * intensity_T - intensity_pB

	# emission ([..., 0]; epsilon(z)) and absorption ([..., 1]; kappa(z)) coefficient per unit volume
	# dtau = - kappa dz
	# I' / I = - kappa dz --> I' emerging intensity; I incident intensity;
	# intensity = torch.exp(raw[..., 0]) * dists # emission per sampled point [n_rays, n_samples]

	# absorption = torch.exp(-nn.functional.relu(raw[..., 1] + noise) * dists) # transmission per sampled point [n_rays, n_samples]
	# [1, .9, 1, 0, 0, 1] --> less dense objects transmit light (1); dense objects absorbe light (0)

	# compute total absorption for each light ray (intensity)
	# how much light is transmitted from each sampled point
	# total_absorption = cumprod_exclusive(absorption + 1e-10) # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)


	
	# [(1), 1, .9, .9, 0, 0] --> total absorption for each point along the ray
	# apply absorption to intensities
	# emerging_intensity = intensity * total_absorption  # integrate total intensity [n_rays, n_samples - 1]

    # intensity (total and polarised) from all electrons
	# for one electron * electron density * weighted by distance along LOS
	emerging_tB = intensity_tB * electron_density * dists
	emerging_pB = intensity_pB * electron_density * dists
	# sum all intensity contributions along LOS
	pixel_tB = emerging_tB.sum(1)[:, None] 
	pixel_pB = emerging_pB.sum(1)[:, None] 

	# target images are already logged
	v_min, v_max = -18, -10
	pixel_tB = (torch.log(pixel_tB) - v_min) / (v_max - v_min) # normalization
	pixel_pB = (torch.log(pixel_pB) - v_min) / (v_max - v_min) # normalization

	# set the weigths to the intensity contributions (sample primary contributing regions)
    # need weights for sampling for fine model
	weights = electron_density / (electron_density.sum(1)[:, None] + 1e-10)

	return pixel_tB, pixel_pB, weights #, absorption


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
				 coarse_model: nn.Module,  
				 fine_model: nn.Module,
				 near: float, 
				 far: float,
				 encoding_fn: Callable[[torch.Tensor], torch.Tensor],
				 sample_stratified: Callable[[torch.Tensor, torch.Tensor, float, float, int], Tuple[torch.Tensor, torch.Tensor]] = sample_non_uniform_box,
				 kwargs_sample_stratified: dict = None, 
				 n_samples_hierarchical: int = 0,
				 kwargs_sample_hierarchical: dict = None):
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

	# vol_render = volume_render()

	# Set no kwargs if none are given.
	if kwargs_sample_stratified is None:
		kwargs_sample_stratified = {}
	if kwargs_sample_hierarchical is None:
		kwargs_sample_hierarchical = {}

	# Sample query points along each ray.
	query_points, z_vals = sample_stratified(
		rays_o, rays_d, near, far, **kwargs_sample_stratified)

	# add time to query points
	exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
	query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

	# Prepare points --> encoding.
	enc_query_points = encoding_fn(query_points_time.view(-1, 4))

	# Coarse model pass.
	raw = coarse_model(enc_query_points)
	raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

	# Perform differentiable volume rendering to re-synthesize the filtergrams.
	channel_map, weights, absorption = raw2outputs(raw, query_points, z_vals, rays_o, rays_d)
	outputs = {'z_vals_stratified': z_vals}

	# Fine model pass.
	if n_samples_hierarchical > 0:
		# Save previous outputs to return.
		channel_map_0 = channel_map

		# Apply hierarchical sampling for fine query points.
		query_points, z_vals_combined, z_hierarch = sample_hierarchical(
			rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
			**kwargs_sample_hierarchical)

		# add time to query points = expand to dimensions of query points and slice one dimension
		exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
		query_points_time = torch.cat([query_points, exp_times], -1)

		# Prepare inputs as before.
		enc_query_points = encoding_fn(query_points_time.view(-1, 4))

		# Forward pass new samples through fine model.
		fine_model = fine_model if fine_model is not None else coarse_model
		raw = fine_model(enc_query_points)
		raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

		# Perform differentiable volume rendering to re-synthesize the filtergrams.
		channel_map, weights, absorption = raw2outputs(raw, query_points, z_vals_combined, rays_o, rays_d)

		# Store outputs.
		outputs['z_vals_hierarchical'] = z_hierarch
		outputs['channel_map_0'] = channel_map_0

	# compute image of absorption
	absorption_map = (1 - absorption).sum(-1)

	# compute regularization of absorption
	distance = query_points.pow(2).sum(-1).pow(0.5)
	regularization = torch.relu(distance - 1.2) * (1 - absorption) # penalize absorption past 1.2 solar radii

	# compute image of height
	height_map = (weights * distance).sum(-1)

	# Store outputs.
	outputs['channel_map'] = channel_map
	outputs['weights'] = weights
	outputs['height_map'] = height_map
	outputs['absorption_map'] = absorption_map
	outputs['regularization'] = regularization
	return outputs