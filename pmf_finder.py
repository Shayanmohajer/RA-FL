import torch
def calculate_pmf(tensor):
	unique_values, counts = torch.unique(tensor, return_counts=True)
	pmf = counts.float() / tensor.numel()
	return pmf


