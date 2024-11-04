import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import sys

class CovarianceNet_64(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=10):
		super(CovarianceNet_64, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                       # 3*84*84
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21
		)
		
		self.covariance = CovaBlock()                        # 1*(441*num_classes)

		self.classifier = nn.Sequential(
			nn.LeakyReLU(0.2, True),
			nn.Dropout(),
			nn.Conv1d(1, 1, kernel_size=256, stride=256, bias=use_bias),
		)


	def forward(self, input1, input2):

		# extract features of input1--query image
		q = self.features(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			S.append(self.features(input2[i]))

		x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
		x = self.classifier(x)    # get Batch*1*num_classes
		x = x.squeeze(1)          # get Batch*num_classes

		return x, x, x



#========================== Define a Covariance Metric layer ==========================#
# Calculate the local covariance matrix of each category in the support set
# Calculate the Covariance Metric between a query sample and a category


class CovaBlock(nn.Module):
	def __init__(self):
		super(CovaBlock, self).__init__()


	# calculate the covariance matrix 
	def cal_covariance(self, input):
		
		CovaMatrix_list = []
		for i in range(len(input)):
			support_set_sam = input[i]
			B, C, h, w = support_set_sam.size()

			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			mean_support = torch.mean(support_set_sam, 1, True)
			support_set_sam = support_set_sam-mean_support

			covariance_matrix = support_set_sam@torch.transpose(support_set_sam, 0, 1)
			covariance_matrix = torch.div(covariance_matrix, h*w*B-1)
			CovaMatrix_list.append(covariance_matrix)

		return CovaMatrix_list    


	# calculate the similarity  
	def cal_similarity(self, input, CovaMatrix_list):
	
		B, C, h, w = input.size()
		Cova_Sim = []
	
		for i in range(B):
			query_sam = input[i]
			query_sam = query_sam.view(C, -1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)    
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				mea_sim = torch.zeros(1, len(CovaMatrix_list)*h*w).cuda()

			for j in range(len(CovaMatrix_list)):
				temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
				mea_sim[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

			Cova_Sim.append(mea_sim.unsqueeze(0))

		Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)
		return Cova_Sim 


	def forward(self, x1, x2):

		CovaMatrix_list = self.cal_covariance(x2)
		Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)

		return Cova_Sim