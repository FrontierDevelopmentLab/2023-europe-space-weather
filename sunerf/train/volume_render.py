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
				v_min: float, v_max: float,
				raw_noise_std: float = 0.0
				) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	r"""
	Convert the raw NeRF output into electron density (1 model output).

	raw: output of NeRF, 2 values per sampled point
	z_vals: distance along the ray as measure from the observer
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
	velocity = raw[:, :, 1:]

    # HOWARD AND TAPPIN 2009 FIG 3
	# working with units of solar radii
    # half angular width of Sun (angle between SQ and ST)
	s_q = query_points.pow(2).sum(-1).pow(0.5)
	s_t = 1
	omega = torch.asin(s_t / s_q)
	
	# z = distance Q to observer
	z = z_vals * torch.norm(rays_d[..., None, :], dim=-1) # distance between observer and scattering point Q

    # chi = scattering angle between line of sight (OS) and QS (dot product)
	chi = torch.acos((rays_d[:, None] * query_points).sum(-1) / (rays_d.pow(2).sum(-1).pow(0.5)[:, None] * query_points.pow(2).sum(-1).pow(0.5) + 1e-6))
	# u = limb darkening coeff 
	# TODO hard coding for now [Ramos 2023]
	u = 0.63

	# sigma_e = thomson cross section [Howard and Tappin 2009 eqn 3]
	r_sun = 6957e+8 # m / solar radii
	sigma_e = 7.95e-30 # m2/sr

	# I0 = intensity of the source (Sun) as a power per unit area (of the photosphere) per unit solid angle
	#    = mean solar radiance ( = irradiance / 4pi)
	# TODO average for now (move to lookup table later)
	I0 = 1361 / 4*torch.pi # W·sr−1·m−2 # (Prša et al., 2016)

	ln = torch.log((1 + torch.sin(omega)) / (torch.cos(omega)))
	cos2_sin = torch.cos(omega) ** 2 / (torch.sin(omega))
	A = torch.cos(omega) * torch.sin(omega) ** 2
	B = - (1/8) * (1 - 3 * torch.sin(omega) ** 2 - cos2_sin * (1 + 3 * torch.sin(omega) ** 2) * ln)
	C = (4 / 3) - torch.cos(omega) - torch.cos(omega) ** 3 / 3
	D = (1 / 8) * (5 + torch.sin(omega) ** 2 - cos2_sin * (5 - torch.sin(omega) ** 2) * ln)

 	# - (1/8) * (1 - 3 * sin(omega) ** 2 - cos(omega) ** 2 / (sin(omega) + 1e-6) * (1 + 3 * sin(omega) ** 2) *  ln((1 + sin(omega)) / (cos(omega) + 1e-6)))

    # equations 23, 24, 29
	intensity_T = I0 * torch.pi * sigma_e  / (2 * z ** 2) * ((1 - u) * C + u * D)
	intensity_pB = I0 * torch.pi * sigma_e / (2 * z ** 2) * torch.sin(chi) ** 2 * ((1 - u) * A + u * B)

	intensity_tB = 2 * intensity_T - intensity_pB

	if torch.isnan(intensity_tB).any() or torch.isnan(intensity_pB).any():
		cond = torch.isnan(intensity_tB) | torch.isnan(intensity_pB)
		print(f'Invalid values in intensity_tB or intensity_pB: query points {query_points[cond]}')
		# remove nan values (where omega close to 0)
		intensity_tB = torch.nan_to_num(intensity_tB, nan=0.0, posinf=0.0, neginf=0.0)
		intensity_pB = torch.nan_to_num(intensity_pB, nan=0.0, posinf=0.0, neginf=0.0)

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
	# for one electron * electron density * weighted by line element ds- separation between sampling points
	emerging_tB = intensity_tB * electron_density * dists
	emerging_pB = intensity_pB * electron_density * dists

	# sum all intensity contributions along LOS
	pixel_tB = emerging_tB.sum(1)[:, None] 
	pixel_pB = emerging_pB.sum(1)[:, None] 
	
	# height and density maps
	# electron_density: (batch, sampling_points, 1), s_q: (batch, sampling_points, 1)
	pixel_density = (electron_density * dists).sum(1)
	height_from_sun = (electron_density * s_q).sum(1) / (electron_density.sum(1) + 1e-10)
	height_from_obs = (electron_density * z).sum(1) / (electron_density.sum(1) + 1e-10)
	
	# clip pixel_tB and pixel_pB to over [e^v_min, e^v_max]
	# device = 'cuda' if torch.cuda.is_available() else "cpu"
	# pixel_tB = torch.clip(pixel_tB, torch.exp(torch.tensor(v_min).to(device)), torch.exp(torch.tensor(v_max).to(device)))
	# pixel_pB = torch.clip(pixel_pB, torch.exp(torch.tensor(v_min).to(device)), torch.exp(torch.tensor(v_max).to(device)))
	# if np.any(pixel_tB.cpu().detach().numpy() < np.exp(v_min)):
	# 	print("Pixel tB smaller than expected: {} < {}".format(pixel_tB.cpu().detach().numpy(), np.exp(v_min)))
	# if np.any(pixel_pB.cpu().detach().numpy() < np.exp(v_min)):
	# 	print("Pixel pB smaller than expected: {} < {}".format(pixel_pB.cpu().detach().numpy(), np.exp(v_min)))
	# pixel_tB = torch.clamp(pixel_tB, min=np.exp(v_min), max=np.exp(v_max))
	# pixel_pB = torch.clamp(pixel_pB, min=np.exp(v_min), max=np.exp(v_max))

	# target images are already logged
	pixel_tB = (torch.log(pixel_tB) - v_min) / (v_max - v_min) # normalization
	pixel_pB = (torch.log(pixel_pB) - v_min) / (v_max - v_min) # normalization
	pixel_B = torch.cat([pixel_tB, pixel_pB], dim=-1)

	# set the weigths to the intensity contributions (sample primary contributing regions)
    # need weights for sampling for fine model
	weights = electron_density / (electron_density.sum(1)[:, None] + 1e-10)

	return pixel_B, pixel_density, height_from_sun, height_from_obs, weights

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
				 vmin: float, 
				 vmax: float,
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
	#print("FIRST LAST INPUTS", rays_o[0], rays_d[0], rays_o[-1], rays_d[-1], times[0], near, far, vmin, vmax, kwargs_sample_stratified, n_samples_hierarchical, kwargs_sample_hierarchical)

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
	pixel_B, pixel_density, height_from_sun, height_from_obs, weights = raw2outputs(raw, query_points, z_vals, rays_o, rays_d, vmin, vmax)
	outputs = {'z_vals_stratified': z_vals}


	
	# Fine model pass.
	if n_samples_hierarchical > 0:
		# Save previous outputs to return.
		pixel_B_0 = pixel_B

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
		pixel_B, pixel_density, height_from_sun, height_from_obs, weights = raw2outputs(raw, query_points, z_vals_combined, rays_o, rays_d, vmin, vmax)

		# Store outputs.
		outputs['z_vals_hierarchical'] = z_hierarch
		outputs['pixel_B_0'] = pixel_B_0

	# compute density and height maps
	density_map = pixel_density
	height_map_sun = height_from_sun
	height_map_obs = height_from_obs

	# compute regularization of absorption
	# distance = query_points.pow(2).sum(-1).pow(0.5)
	# regularization = torch.relu(distance - 1.2) * (1 - absorption) # penalize absorption past 1.2 solar radii

	# compute image of height
	# height_map = (weights * distance).sum(-1)

	# Store outputs.
	outputs['pixel_B'] = pixel_B
	outputs['weights'] = weights
	outputs['height_map_sun'] = height_map_sun
	outputs['height_map_obs'] = height_map_obs
	outputs['density_map'] = density_map
	# outputs['regularization'] = regularization
	return outputs


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix