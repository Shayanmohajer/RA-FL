

from pmf_finder import calculate_pmf
from huffman_gen import huffman_coding
import torch
import numpy as np

def total_length(model):
	layer_count = 0
	tot_length = 0
	for name, param in model.named_parameters():
		if 'weight' in name and param.data.dim()>=4:
			weight = param.data
			AVG_length = huffman_coding(calculate_pmf(weight))
			tot_length = tot_length + AVG_length
			layer_count+=1
	return tot_length/layer_count


q_weights = {}
def total_length2(model): #when LSQ is used
	layer_count = 0
	tot_length = 0

	for name, module in model.named_modules():
		if hasattr(module, 'weight'):
			if len(module.weight.size())>=4:
				layer_count+=1
				if layer_count>1: 
					indices = module.quan_w_fn.quantized_x.detach()
					q = module.quan_w_fn.s_scale.detach()
					q_weights[name] = indices.unique() * q
					AVG_length = huffman_coding(calculate_pmf(indices))
					tot_length = tot_length + AVG_length
					N_layers = layer_count
					# print(q)
	# breakpoint()
	return tot_length/layer_count, q_weights, N_layers


q_weights = {}
def quant_weight(model): #when LSQ is used
	layer_count = 0
	tot_length = 0
	for name, module in model.named_modules():
		if hasattr(module, 'weight'):
			if len(module.weight.size())>=4:
				layer_count+=1
				if layer_count>1: 
					indices = module.quan_w_fn.quantized_x.detach()
					q = module.quan_w_fn.s_scale.detach()
					q_weights[name] = indices.unique() * q
	return q_weights

def curves(model, args): 
	layer_count = 0
	for name, module in model.named_modules():
		if hasattr(module, 'weight'):
			if len(module.weight.view(-1))>=1e4:
				indices = module.quan_w_fn.quantized_x.detach()
				q = module.quan_w_fn.s_scale.detach()
				quant = (indices * q).view(-1)
				quant = quant.cpu().detach().numpy()
				fp = (module.weight).view(-1)
				fp = fp.cpu().detach().numpy()
				# breakpoint()
				np.save('T='+str(args.temp)+'b=' + str(args.weightBitwidth)+ 'fp.npy', fp)
				np.save('T='+str(args.temp)+'b=' + str(args.weightBitwidth)+ 'quant.npy', quant)
				break
	return fp,quant
