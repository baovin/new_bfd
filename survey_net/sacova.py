import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import pdb
import sys

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x
    
#--------Multi-head attention Module-----------------------------------------------------------------------#
class MHSA(nn.Module):
    def __init__(self, n_dims = 64, width=16, height=16, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm([n_dims, width, height])

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        out = self.norm(out) + x
        return out   


class SA_CovaMNET(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=10):
        super(SA_CovaMNET, self).__init__()

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

        self.mhsa = MHSA(n_dims=64, width=16, height=16, heads=4)
        self.cbam = CBAM(channels=64, r=16)
        self.covariance = CovaBlock()  

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=256, stride=256, bias=use_bias),
        )

    def forward(self, input1, input2):

      
        features = self.features(input1)
        q = self.mhsa(features) + self.cbam(features) + features


        S = []
        for i in range(len(input2)):
            features_s = self.features(input2[i])
            S.append(self.mhsa(features_s) + self.cbam(features_s) + features_s)

        x = self.covariance(q, S)  
        x = self.classifier(x)     
        x = x.squeeze(1)          

        return x, x, x


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