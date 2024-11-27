import torch
import torch.nn as nn
import numpy as np
import itertools



def pwlq(model, num_bits=4):
	for layer in model.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			print("*"*20)
			weights = layer.weight.data

			# 1. Distribution Analysis
			hist, bin_edges = np.histogram(weights.detach().cpu().numpy(), bins='auto')
			bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

			# 2. Breakpoint Calculation
			breakpoints = optimal_breakpoints(bin_centers, hist, num_levels=2**num_bits)

			# 3. Quantization Mapping
			def quantize_tensor(tensor):
				quantized = torch.bucketize(tensor, torch.from_numpy(breakpoints))
				return quantized.float()  # Maintain float type

			# 4. Quantize and Update Weights
			layer.weight.data = quantize_tensor(weights)

def optimal_breakpoints(bin_centers, hist, num_levels):
    """Approximates optimal breakpoints using grid search."""

    min_error = float('inf')
    best_breakpoints = None

    # Dense grid around data center
    grid_min = bin_centers.min() - 0.2 * bin_centers.std()  
    grid_max = bin_centers.max() + 0.2 * bin_centers.std()  
    grid = np.linspace(grid_min, grid_max, num=50)  # Adjust granularity

    # Test different breakpoint combinations from grid
    for breaks in itertools.combinations(grid, num_levels - 1):
        breaks = sorted(breaks)  # Ensure order
        error = quantization_error(bin_centers, hist, breaks) 
        if error < min_error:
            min_error = error
            best_breakpoints = breaks

    return np.array(best_breakpoints)

def quantization_error(bin_centers, hist, breakpoints):
    """Calculates the quantization error for a given set of breakpoints.
       Approximates error based on squared differences and distribution.
    """

    breakpoints = np.array([-np.inf] + list(breakpoints) + [np.inf])  # Add boundaries
    quantized_values = np.digitize(bin_centers, breakpoints, right=False) - 1  # Assign bin indices

    # Calculate representative value for each bin
    bin_values = np.array([bin_centers[quantized_values == i].mean() for i in range(len(breakpoints) - 1)])

    # Squared error weighted by histogram
    error = np.sum(((bin_centers - bin_values) ** 2) * hist) 

    return error


