import torch
import torch.nn as nn
#-----------------------Convmixer-----------------------------------------------#
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(inplane = 1, dim = 64, depth=1, kernel_size=64, patch_size=4):
    return nn.Sequential(
        nn.Conv2d(inplane , dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, (1, kernel_size), groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, (kernel_size, 1), groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)]
    )

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
    


import functools

class MFNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=7):
        super(MFNet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.mixer = ConvMixer()
        self.MHSA = MHSA()

        self.covariance = CovaBlock()

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=256, stride=256, bias=use_bias),
        )


    def forward(self, input1, input2):
        #q = self.features(input1)
        q = self.MHSA(self.mixer(input1))
        print(q.shape)
        S = []
        for i in range(len(input2)):
            features = self.mixer(input2[i])
            print(features.shape)
            S.append(self.MHSA(features))
        x = self.covariance(q, S)

        x = self.classifier(x.view(x.size(0), 1, -1))

        x = x.squeeze(1)


        return x

#-----------------------MF-Net-----------------------------------------------#

class CovaBlock(nn.Module):
    def __init__(self):
        super(CovaBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h * w * B - 1)
            CovaMatrix_list.append(covariance_matrix)
        return CovaMatrix_list

    def cal_similarity(self, input, CovaMatrix_list):
        B, C, h, w = input.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            if torch.cuda.is_available():
                mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            else:
                mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w)
            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.view(1, -1))

        Cova_Sim = torch.cat(Cova_Sim, 0)

        return Cova_Sim

    def forward(self, x1, x2):
        CovaMatrix_list = self.cal_covariance(x2)
        Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)

        return Cova_Sim
   