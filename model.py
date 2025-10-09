import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
import basic_net as basic_net
import yaml
import os
from yaml.loader import SafeLoader
from net import *
import copy
from torch.nn import CrossEntropyLoss, Dropout

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from utils import FocalLoss
import scipy.sparse as sp
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class  CNN_cls(nn.Module):
    def __init__(self, opt):
        super(CNN_cls, self).__init__()
        self.opt = opt
        self.Name = self.opt['Clstype']
        self.encoder_norm_endtoend = LayerNorm(1024, eps=1e-6)

        self.attention = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)

        if self.Name=='Diag':
            self.n_classes= 6
            self.fc = nn.Linear(1024, self.n_classes)
            if self.opt['TrainingSet']=='TCGA':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([0.55,4,9,6,7,5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            elif self.opt['TrainingSet']=='All':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([1, 6.7, 19, 14, 11, 9])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            #[0.01, 10, 40, 40, 10, 10]
        elif self.Name=='DiagSim':
            self.n_classes = 4
            self.fc = nn.Linear(1024, self.n_classes)
            if self.opt['TrainingSet'] == 'TCGA':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([0.3, 4, 3.5,2.8])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            elif self.opt['TrainingSet'] == 'All':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([0.9,6.7,8,5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
                    # [0.1,5,6.8,4.8]
    def forward(self,x):
        """
            x: [BS,2500,1024]
        """
        # hidden_states = self.encoder_norm_endtoend(x)  # [B,2500,1024]

        # A_V = self.attention_V(x)  # BxNx128
        # A_U = self.attention_U(x)  # BxNx128
        # A_encoded = self.attention_weights(A_V * A_U)  # BxNx1
        # A_encoded = F.softmax(A_encoded, dim=1)[..., 0]  # BxN AMIL attention map

        # A_encoded = self.attention(x)[..., 0]  # BxN
        A_encoded=torch.mean(x,dim=2)
        for i in range(x.shape[0]):
            if i == 0:
                Final_con_layer = x[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, x[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_His_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = x[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, x[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.cat((Final_con_layer_His, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_His_new = torch.cat(
                    (encoded_His_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded = encoded_His_new  # Bx1024

        ####################saliency maps for G4GBM
        weight_G4GBM = torch.unsqueeze(self.fc.weight[0], dim=1)  # [512,1]
        saliency_G4GBM = torch.matmul(Final_con_layer_His, weight_G4GBM)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G4GBM = saliency_G4GBM + self.fc.bias[0] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G4A
        weight_G4A = torch.unsqueeze(self.fc.weight[1], dim=1)  # [512,1]
        saliency_G4A = torch.matmul(Final_con_layer_His, weight_G4A)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G4A = saliency_G4A + self.fc.bias[1] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G3A
        weight_G3A = torch.unsqueeze(self.fc.weight[2], dim=1)  # [512,1]
        saliency_G3A = torch.matmul(Final_con_layer_His, weight_G3A)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G3A = saliency_G3A + self.fc.bias[2] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G2A
        weight_G2A = torch.unsqueeze(self.fc.weight[3], dim=1)  # [512,1]
        saliency_G2A = torch.matmul(Final_con_layer_His, weight_G2A)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G2A = saliency_G2A + self.fc.bias[3] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G3O
        weight_G3O = torch.unsqueeze(self.fc.weight[4], dim=1)  # [512,1]
        saliency_G3O = torch.matmul(Final_con_layer_His, weight_G3O)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G3O = saliency_G3O + self.fc.bias[4] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G2O
        weight_G2O = torch.unsqueeze(self.fc.weight[5], dim=1)  # [512,1]
        saliency_G2O = torch.matmul(Final_con_layer_His, weight_G2O)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G2O = saliency_G2O + self.fc.bias[5] / encoded.shape[1]  # [BSxN

        logits = self.fc(encoded)  # [BS,cls]
        results_dict = {'logits': logits}
        # return results_dict, saliency_G4GBM, saliency_G4A, saliency_G3A, saliency_G2A, saliency_G3O, saliency_G2O
        return results_dict

        # ####################saliency maps for G4GBM
        # weight_G4GBM = torch.unsqueeze(self.fc.weight[0], dim=1)  # [512,1]
        # saliency_G4GBM = torch.matmul(Final_con_layer_His, weight_G4GBM)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G4GBM = saliency_G4GBM + self.fc.bias[0] / encoded.shape[1]  # [BSxN]
        #
        # ####################saliency maps for G4A
        # weight_G4A = torch.unsqueeze(self.fc.weight[1], dim=1)  # [512,1]
        # saliency_G4A = torch.matmul(Final_con_layer_His, weight_G4A)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G4A = saliency_G4A + self.fc.bias[1] / encoded.shape[1]  # [BSxN]
        #
        # ####################saliency maps for G23A
        # weight_G23A = torch.unsqueeze(self.fc.weight[2], dim=1)  # [512,1]
        # saliency_G23A = torch.matmul(Final_con_layer_His, weight_G23A)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G23A = saliency_G23A + self.fc.bias[2] / encoded.shape[1]  # [BSxN]
        #
        # ####################saliency maps for G23O
        # weight_G23O= torch.unsqueeze(self.fc.weight[3], dim=1)  # [512,1]
        # saliency_G23O = torch.matmul(Final_con_layer_His, weight_G23O)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G23O = saliency_G23O + self.fc.bias[3] / encoded.shape[1]  # [BSxN]
        #
        # logits = self.fc(encoded)  # [BS,cls]
        # results_dict = {'logits': logits}
        # # return results_dict, saliency_G4GBM, saliency_G4A, saliency_G23A, saliency_G23O
        # return results_dict




    def calculateLoss(self, pred0, GT):
        FLAT_normal = False
        self.loss = 0
        count = 0
        NA_cls= 4 if self.Name=='DiagSim' else 6
        for i in range(GT.detach().cpu().numpy().shape[0]):
            if GT.detach().cpu().numpy()[i] != NA_cls:
                if count == 0:
                    pred = pred0[i].unsqueeze(0)
                    label = GT[i].unsqueeze(0)
                    count += 1
                else:
                    pred = torch.cat((pred, pred0[i].unsqueeze(0)), 0)
                    label = torch.cat((label, GT[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss = 0
        else:
            self.loss = self.criterion_ce(pred, label)
        return self.loss


class  CNN_cls_marker(nn.Module):
    def __init__(self, opt):
        super(CNN_cls_marker, self).__init__()
        self.opt = opt
        self.encoder_norm_endtoend = LayerNorm(1024, eps=1e-6)
        self.marker=self.opt['name'].split('_')[2]

        self.n_classes= 2
        self.fc = nn.Linear(1024, self.n_classes)
        if self.marker=='IDH':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 6.7])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.marker == '1p19q':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 4])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.marker == 'CDKN':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])

    def forward(self,x):
        """
            x: [BS,2500,1024]
        """
        A_encoded=torch.mean(x,dim=2)
        for i in range(x.shape[0]):
            if i == 0:
                Final_con_layer = x[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, x[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_His_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = x[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, x[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.cat((Final_con_layer_His, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_His_new = torch.cat(
                    (encoded_His_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded = encoded_His_new  # Bx1024
        logits = self.fc(encoded)  # [BS,cls]
        results_dict = {'logits': logits}
        return results_dict



    def calculateLoss(self, pred0, GT):
        FLAT_normal = False
        self.loss = 0
        count = 0
        NA_cls= 2
        for i in range(GT.detach().cpu().numpy().shape[0]):
            if GT.detach().cpu().numpy()[i] != NA_cls:
                if count == 0:
                    pred = pred0[i].unsqueeze(0)
                    label = GT[i].unsqueeze(0)
                    count += 1
                else:
                    pred = torch.cat((pred, pred0[i].unsqueeze(0)), 0)
                    label = torch.cat((label, GT[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss = 0
        else:
            self.loss = self.criterion_ce(pred, label)
        return self.loss

class  CNN_cls_predall(nn.Module):
    def __init__(self, opt):
        super(CNN_cls_predall, self).__init__()
        self.opt = opt

        self.n_classes = 2
        if self.opt['TrainingSet'] == 'All':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.6])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_CDKN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.3, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_MGMT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_710 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_ATRX = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_EGFR = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.7, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_TERT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_OLIG2 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([45, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PDGFRA = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 10])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PTEN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 3.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_P53 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.opt['TrainingSet'] == 'TCGA':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 4.4])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_CDKN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_MGMT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_710 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_ATRX = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 3])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_EGFR = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_TERT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PDGFRA = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 10])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PTEN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 6.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_P53 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.6])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])

        elif self.opt['TrainingSet'] == 'Tiantan':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.4, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_MGMT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([4, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_ATRX = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_EGFR = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([9, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PTEN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_P53 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_OLIG2 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([45, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])


        self._fc2_IDH_1 = nn.Linear(1024, self.n_classes)
        self._fc2_1p19q_1 = nn.Linear(1024, self.n_classes)
        self._fc2_ATRX_1 = nn.Linear(1024, self.n_classes)
        self._fc2_EGFR_1 = nn.Linear(1024, self.n_classes)
        self._fc2_MGMT_1 = nn.Linear(1024, self.n_classes)
        self._fc2_PTEN_1 = nn.Linear(1024, self.n_classes)
        self._fc2_P53_1 = nn.Linear(1024, self.n_classes)

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            self._fc2_CDKN_1 = nn.Linear(1024, self.n_classes)
            self._fc2_710_1 = nn.Linear(1024, self.n_classes)
            self._fc2_TERT_1 = nn.Linear(1024, self.n_classes)
            self._fc2_PDGFRA_1 = nn.Linear(1024, self.n_classes)
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            self._fc2_OLIG2_1 = nn.Linear(1024, self.n_classes)

        self.encoder_norm_IDH = LayerNorm(1024, eps=1e-6)
        self.encoder_norm_1p19q = LayerNorm(1024, eps=1e-6)
        self.encoder_norm_MGMT = LayerNorm(1024, eps=1e-6)
        self.encoder_norm_ATRX = LayerNorm(1024, eps=1e-6)
        self.encoder_norm_EGFR = LayerNorm(1024, eps=1e-6)
        self.encoder_norm_PTEN = LayerNorm(1024, eps=1e-6)
        self.encoder_norm_P53 = LayerNorm(1024, eps=1e-6)

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            self.encoder_norm_CDKN = LayerNorm(1024, eps=1e-6)
            self.encoder_norm_710 = LayerNorm(1024, eps=1e-6)
            self.encoder_norm_TERT = LayerNorm(1024, eps=1e-6)
            self.encoder_norm_PDGFRA = LayerNorm(1024, eps=1e-6)
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            self.encoder_norm_OLIG2 = LayerNorm(1024, eps=1e-6)



    def forward(self,x):
        """
            x: [BS,2500,1024]
        """

        ########################   IDH   ########################
        encoded_IDH = self.encoder_norm_IDH(x)  # [BS,2500,512]
        A_encoded_IDH = F.softmax(encoded_IDH, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_IDH.shape[0]):
            if i == 0:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_IDH_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.cat((Final_con_layer_IDH, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_IDH_new = torch.cat(
                    (encoded_IDH_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_IDH = encoded_IDH_new  # Bx512
        ########################   1p19q   ########################
        encoded_1p19q = self.encoder_norm_1p19q(x)  # [BS,2500,512]
        A_encoded_1p19q = F.softmax(encoded_1p19q, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_1p19q.shape[0]):
            if i == 0:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_1p19q_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.cat((Final_con_layer_1p19q, torch.unsqueeze(Final_con_layer, 0)),
                                                  dim=0)  # BSxNx512
                encoded_1p19q_new = torch.cat(
                    (encoded_1p19q_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_1p19q = encoded_1p19q_new  # Bx512

        ########################   MGMT   ########################
        encoded_MGMT = self.encoder_norm_MGMT(x)  # [BS,2500,512]
        A_encoded_MGMT = F.softmax(encoded_MGMT, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_MGMT.shape[0]):
            if i == 0:
                Final_con_layer = encoded_MGMT[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_MGMT[i], 1).expand(-1, encoded_MGMT[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_MGMT = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_MGMT_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_MGMT[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_MGMT[i], 1).expand(-1, encoded_MGMT[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_MGMT = torch.cat((Final_con_layer_MGMT, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_MGMT_new = torch.cat(
                    (encoded_MGMT_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_MGMT = encoded_MGMT_new  # Bx512

        ########################   ATRX   ########################
        encoded_ATRX = self.encoder_norm_ATRX(x)  # [BS,2500,512]
        A_encoded_ATRX = F.softmax(encoded_ATRX, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_ATRX.shape[0]):
            if i == 0:
                Final_con_layer = encoded_ATRX[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_ATRX[i], 1).expand(-1, encoded_ATRX[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_ATRX = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_ATRX_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_ATRX[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_ATRX[i], 1).expand(-1, encoded_ATRX[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_ATRX = torch.cat((Final_con_layer_ATRX, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_ATRX_new = torch.cat(
                    (encoded_ATRX_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_ATRX = encoded_ATRX_new  # Bx512

        ########################   EGFR   ########################
        encoded_EGFR = self.encoder_norm_EGFR(x)  # [BS,2500,512]
        A_encoded_EGFR = F.softmax(encoded_EGFR, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_EGFR.shape[0]):
            if i == 0:
                Final_con_layer = encoded_EGFR[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_EGFR[i], 1).expand(-1, encoded_EGFR[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_EGFR = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_EGFR_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_EGFR[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_EGFR[i], 1).expand(-1, encoded_EGFR[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_EGFR = torch.cat((Final_con_layer_EGFR, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_EGFR_new = torch.cat(
                    (encoded_EGFR_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_EGFR = encoded_EGFR_new  # Bx512
        ########################   PTEN   ########################
        encoded_PTEN = self.encoder_norm_PTEN(x)  # [BS,2500,512]
        A_encoded_PTEN = F.softmax(encoded_PTEN, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_PTEN.shape[0]):
            if i == 0:
                Final_con_layer = encoded_PTEN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_PTEN[i], 1).expand(-1, encoded_PTEN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_PTEN = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_PTEN_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_PTEN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_PTEN[i], 1).expand(-1, encoded_PTEN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_PTEN = torch.cat((Final_con_layer_PTEN, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_PTEN_new = torch.cat(
                    (encoded_PTEN_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_PTEN = encoded_PTEN_new  # Bx512
        ########################   P53   ########################
        encoded_P53 = self.encoder_norm_P53(x)  # [BS,2500,512]
        A_encoded_P53 = F.softmax(encoded_P53, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_P53.shape[0]):
            if i == 0:
                Final_con_layer = encoded_P53[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_P53[i], 1).expand(-1, encoded_P53[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_P53 = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_P53_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_P53[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_P53[i], 1).expand(-1, encoded_P53[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_P53 = torch.cat((Final_con_layer_P53, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_P53_new = torch.cat(
                    (encoded_P53_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_P53 = encoded_P53_new  # Bx512
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            ########################   CDKN   ########################
            encoded_CDKN = self.encoder_norm_CDKN(x)  # [BS,2500,512]
            A_encoded_CDKN = F.softmax(encoded_CDKN, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_CDKN.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_CDKN[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_CDKN = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_CDKN_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_CDKN[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_CDKN = torch.cat((Final_con_layer_CDKN, torch.unsqueeze(Final_con_layer, 0)),
                                                     dim=0)  # BSxNx512
                    encoded_CDKN_new = torch.cat(
                        (encoded_CDKN_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_CDKN = encoded_CDKN_new  # Bx512
            ########################   710   ########################
            encoded_710 = self.encoder_norm_710(x)  # [BS,2500,512]
            A_encoded_710 = F.softmax(encoded_710, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_710.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_710[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_710[i], 1).expand(-1, encoded_710[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_710 = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_710_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_710[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_710[i], 1).expand(-1, encoded_710[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_710 = torch.cat((Final_con_layer_710, torch.unsqueeze(Final_con_layer, 0)),
                                                    dim=0)  # BSxNx512
                    encoded_710_new = torch.cat(
                        (encoded_710_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_710 = encoded_710_new  # Bx512
            ########################   TERT   ########################
            encoded_TERT = self.encoder_norm_TERT(x)  # [BS,2500,512]
            A_encoded_TERT = F.softmax(encoded_TERT, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_TERT.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_TERT[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_TERT[i], 1).expand(-1, encoded_TERT[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_TERT = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_TERT_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_TERT[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_TERT[i], 1).expand(-1, encoded_TERT[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_TERT = torch.cat((Final_con_layer_TERT, torch.unsqueeze(Final_con_layer, 0)),
                                                     dim=0)  # BSxNx512
                    encoded_TERT_new = torch.cat(
                        (encoded_TERT_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_TERT = encoded_TERT_new  # Bx512
            ########################   PDGFRA   ########################
            encoded_PDGFRA= self.encoder_norm_PDGFRA(x)  # [BS,2500,512]
            A_encoded_PDGFRA = F.softmax(encoded_PDGFRA, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_PDGFRA.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_PDGFRA[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_PDGFRA[i], 1).expand(-1,
                                                                                  encoded_PDGFRA[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_PDGFRA = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_PDGFRA_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_PDGFRA[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_PDGFRA[i], 1).expand(-1,
                                                                                  encoded_PDGFRA[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_PDGFRA = torch.cat((Final_con_layer_PDGFRA, torch.unsqueeze(Final_con_layer, 0)),
                                                       dim=0)  # BSxNx512
                    encoded_PDGFRA_new = torch.cat(
                        (encoded_PDGFRA_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_PDGFRA = encoded_PDGFRA_new  # Bx512

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            ########################   OLIG2   ########################
            encoded_OLIG2 = self.encoder_norm_OLIG2(x)  # [BS,2500,512]
            A_encoded_OLIG2 = F.softmax(encoded_OLIG2, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_OLIG2.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_OLIG2[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_OLIG2[i], 1).expand(-1, encoded_OLIG2[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_OLIG2 = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_OLIG2_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_OLIG2[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_OLIG2[i], 1).expand(-1, encoded_OLIG2[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_OLIG2 = torch.cat((Final_con_layer_OLIG2, torch.unsqueeze(Final_con_layer, 0)),
                                                      dim=0)  # BSxNx512
                    encoded_OLIG2_new = torch.cat(
                        (encoded_OLIG2_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_OLIG2 = encoded_OLIG2_new  # Bx512

        logits_IDH = self._fc2_IDH_1(encoded_IDH)  # [BS,2]
        logits_1p19q = self._fc2_1p19q_1(encoded_1p19q)  # [BS,2]
        logits_MGMT = self._fc2_MGMT_1(encoded_MGMT)  # [BS,2]
        logits_ATRX = self._fc2_ATRX_1(encoded_ATRX)  # [BS,2]
        logits_EGFR = self._fc2_EGFR_1(encoded_EGFR)  # [BS,2]
        logits_PTEN = self._fc2_PTEN_1(encoded_PTEN)  # [BS,2]
        logits_P53 = self._fc2_P53_1(encoded_P53)  # [BS,2]
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            logits_CDKN = self._fc2_CDKN_1(encoded_CDKN)  # [BS,2]
            logits_710 = self._fc2_710_1(encoded_710)  # [BS,2]
            logits_TERT = self._fc2_TERT_1(encoded_TERT)  # [BS,2]
            logits_PDGFRA = self._fc2_PDGFRA_1(encoded_PDGFRA)  # [BS,2]

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            logits_OLIG2 = self._fc2_OLIG2_1(encoded_OLIG2)  # [BS,2]

        if self.opt['TrainingSet'] == 'All':
            results_dict = {'logits_IDH': logits_IDH, 'logits_1p19q': logits_1p19q, 'logits_CDKN': logits_CDKN,
                            'logits_MGMT': logits_MGMT,
                            'logits_ATRX': logits_ATRX, 'logits_EGFR': logits_EGFR, 'logits_PTEN': logits_PTEN,
                            'logits_P53': logits_P53,
                            'logits_710': logits_710, 'logits_TERT': logits_TERT, 'logits_PDGFRA': logits_PDGFRA,
                            'logits_OLIG2': logits_OLIG2}

        if self.opt['TrainingSet'] == 'TCGA':
            results_dict = {'logits_IDH': logits_IDH, 'logits_1p19q': logits_1p19q, 'logits_CDKN': logits_CDKN,
                            'logits_MGMT': logits_MGMT,
                            'logits_ATRX': logits_ATRX, 'logits_EGFR': logits_EGFR, 'logits_PTEN': logits_PTEN,
                            'logits_P53': logits_P53,
                            'logits_710': logits_710, 'logits_TERT': logits_TERT, 'logits_PDGFRA': logits_PDGFRA}
        if self.opt['TrainingSet'] == 'Tiantan':
            results_dict = {'logits_IDH': logits_IDH, 'logits_1p19q': logits_1p19q, 'logits_MGMT': logits_MGMT,
                            'logits_ATRX': logits_ATRX,
                            'logits_EGFR': logits_EGFR, 'logits_PTEN': logits_PTEN, 'logits_P53': logits_P53,
                            'logits_OLIG2': logits_OLIG2}

        return results_dict

    def calculateLoss_IDH(self, pred, label):
        FLAT_normal = False
        self.loss_IDH = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_IDH = pred[i].unsqueeze(0)
                    label_IDH = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_IDH = torch.cat((pred_IDH, pred[i].unsqueeze(0)), 0)
                    label_IDH = torch.cat((label_IDH, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_IDH = 0
        else:
            self.loss_IDH = self.criterion_ce_IDH(pred_IDH, label_IDH)
        return self.loss_IDH

    def calculateLoss_1p19q(self, pred, label):
        FLAT_normal = False
        self.loss_1p19q = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_1p19q = pred[i].unsqueeze(0)
                    label_1p19q = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_1p19q = torch.cat((pred_1p19q, pred[i].unsqueeze(0)), 0)
                    label_1p19q = torch.cat((label_1p19q, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_1p19q = 0
        else:
            self.loss_1p19q = self.criterion_ce_1p19q(pred_1p19q, label_1p19q)
        return self.loss_1p19q

    def calculateLoss_CDKN(self, pred, label):
        FLAT_normal = False
        self.loss_CDKN = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_CDKN = pred[i].unsqueeze(0)
                    label_CDKN = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_CDKN = torch.cat((pred_CDKN, pred[i].unsqueeze(0)), 0)
                    label_CDKN = torch.cat((label_CDKN, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_CDKN = 0
        else:
            self.loss_CDKN = self.criterion_ce_CDKN(pred_CDKN, label_CDKN)
        return self.loss_CDKN

    def calculateLoss_MGMT(self, pred, label):
        FLAT_normal = False
        self.loss_MGMT = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_MGMT = pred[i].unsqueeze(0)
                    label_MGMT = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_MGMT = torch.cat((pred_MGMT, pred[i].unsqueeze(0)), 0)
                    label_MGMT = torch.cat((label_MGMT, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_MGMT = 0
        else:
            self.loss_MGMT = self.criterion_ce_MGMT(pred_MGMT, label_MGMT)
        return self.loss_MGMT

    def calculateLoss_ATRX(self, pred, label):
        FLAT_normal = False
        self.loss_ATRX = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_ATRX = pred[i].unsqueeze(0)
                    label_ATRX = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_ATRX = torch.cat((pred_ATRX, pred[i].unsqueeze(0)), 0)
                    label_ATRX = torch.cat((label_ATRX, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_ATRX = 0
        else:
            self.loss_ATRX = self.criterion_ce_ATRX(pred_ATRX, label_ATRX)
        return self.loss_ATRX

    def calculateLoss_EGFR(self, pred, label):
        FLAT_normal = False
        self.loss_EGFR = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_EGFR = pred[i].unsqueeze(0)
                    label_EGFR = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_EGFR = torch.cat((pred_EGFR, pred[i].unsqueeze(0)), 0)
                    label_EGFR = torch.cat((label_EGFR, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_EGFR = 0
        else:
            self.loss_EGFR = self.criterion_ce_EGFR(pred_EGFR, label_EGFR)
        return self.loss_EGFR

    def calculateLoss_PTEN(self, pred, label):
        FLAT_normal = False
        self.loss_PTEN = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_PTEN = pred[i].unsqueeze(0)
                    label_PTEN = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_PTEN = torch.cat((pred_PTEN, pred[i].unsqueeze(0)), 0)
                    label_PTEN = torch.cat((label_PTEN, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_PTEN = 0
        else:
            self.loss_PTEN = self.criterion_ce_PTEN(pred_PTEN, label_PTEN)
        return self.loss_PTEN

    def calculateLoss_TERT(self, pred, label):
        FLAT_normal = False
        self.loss_TERT = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_TERT = pred[i].unsqueeze(0)
                    label_TERT = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_TERT = torch.cat((pred_TERT, pred[i].unsqueeze(0)), 0)
                    label_TERT = torch.cat((label_TERT, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_TERT = 0
        else:
            self.loss_TERT = self.criterion_ce_TERT(pred_TERT, label_TERT)
        return self.loss_TERT

    def calculateLoss_P53(self, pred, label):
        FLAT_normal = False
        self.loss_P53 = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_P53 = pred[i].unsqueeze(0)
                    label_P53 = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_P53 = torch.cat((pred_P53, pred[i].unsqueeze(0)), 0)
                    label_P53 = torch.cat((label_P53, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_P53 = 0
        else:
            self.loss_P53 = self.criterion_ce_P53(pred_P53, label_P53)
        return self.loss_P53

    def calculateLoss_710(self, pred, label):
        FLAT_normal = False
        self.loss_710 = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_710 = pred[i].unsqueeze(0)
                    label_710 = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_710 = torch.cat((pred_710, pred[i].unsqueeze(0)), 0)
                    label_710 = torch.cat((label_710, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_710 = 0
        else:
            self.loss_710 = self.criterion_ce_710(pred_710, label_710)
        return self.loss_710

    def calculateLoss_PDGFRA(self, pred, label):
        FLAT_normal = False
        self.loss_PDGFRA = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_PDGFRA = pred[i].unsqueeze(0)
                    label_PDGFRA = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_PDGFRA = torch.cat((pred_PDGFRA, pred[i].unsqueeze(0)), 0)
                    label_PDGFRA = torch.cat((label_PDGFRA, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_PDGFRA = 0
        else:
            self.loss_PDGFRA = self.criterion_ce_PDGFRA(pred_PDGFRA, label_PDGFRA)
        return self.loss_PDGFRA

    def calculateLoss_OLIG2(self, pred, label):
        FLAT_normal = False
        self.loss_OLIG2 = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_OLIG2 = pred[i].unsqueeze(0)
                    label_OLIG2 = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_OLIG2 = torch.cat((pred_OLIG2, pred[i].unsqueeze(0)), 0)
                    label_OLIG2 = torch.cat((label_OLIG2, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_OLIG2 = 0
        else:
            self.loss_OLIG2 = self.criterion_ce_OLIG2(pred_OLIG2, label_OLIG2)
        return self.loss_OLIG2



    def calculateLoss(self, pred0, GT):
        FLAT_normal = False
        self.loss = 0
        count = 0
        NA_cls= 4 if self.Name=='DiagSim' else 6
        for i in range(GT.detach().cpu().numpy().shape[0]):
            if GT.detach().cpu().numpy()[i] != NA_cls:
                if count == 0:
                    pred = pred0[i].unsqueeze(0)
                    label = GT[i].unsqueeze(0)
                    count += 1
                else:
                    pred = torch.cat((pred, pred0[i].unsqueeze(0)), 0)
                    label = torch.cat((label, GT[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss = 0
        else:
            self.loss = self.criterion_ce(pred, label)
        return self.loss



class Mine_init(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_init, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.default_patchnum = self.opt['fixdim']

        self.Pacth_embedding = nn.Linear(self.size[0], self.size[1])
        self.Pacth_embedding_relu = nn.Sequential(nn.Linear(self.size[0], self.size[1]), nn.ReLU())
        self.Pacth_embedding_2048 = nn.Linear(self.size[0], self.size[1])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.size[1] ))
        self.position_embeddings_clstoken = torch.tensor(nn.Parameter(torch.zeros(1, self.default_patchnum + 1, self.size[1])),
            device='cpu')
        self.position_embeddings = torch.nn.Parameter(torch.FloatTensor(1, self.default_patchnum, self.size[1]))


        self.Dropout=nn.Dropout(0.25)

    def forward(self,x):
        """
            x: [BS, N, 1024 or 2048]
        """
        embeddings = self.Pacth_embedding_relu(x)  # [B, n, 512]
        embeddings = self.Dropout(embeddings)  # [B, n, 512]
        return embeddings

class Mine_endtoend_body(nn.Module):
    def __init__(self, opt):
        super(Mine_endtoend_body, self).__init__()
        self.opt = opt
        self.size = [1024, 512]
        self.layer_endtoend = nn.ModuleList()
        for _ in range(self.opt['Network']['Body_layers']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer_endtoend.append(copy.deepcopy(layer))
        self.encoder_norm_endtoend = LayerNorm(self.size[1], eps=1e-6)

    def forward(self, hidden_states):
        for layer_block in self.layer_endtoend:
            hidden_states, weights = layer_block(hidden_states)
        hidden_states = self.encoder_norm_endtoend(hidden_states)  # [B,2500,512]
        return hidden_states

class Mine_molecular(nn.Module):
    def __init__(self, opt):
        super(Mine_molecular, self).__init__()
        self.opt = opt
        self.size = [1024, 512]
        self.layer_share = nn.ModuleList()
        for _ in range(self.opt['Network']['Mole_layers_share']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer_share.append(copy.deepcopy(layer))

        self.layer_IDH = nn.ModuleList()
        for _ in range(self.opt['Network']['Mole_layers_unique']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer_IDH.append(copy.deepcopy(layer))
        self.layer_pq = nn.ModuleList()
        for _ in range(self.opt['Network']['Mole_layers_unique']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer_pq.append(copy.deepcopy(layer))
        self.layer_CDKN = nn.ModuleList()
        for _ in range(self.opt['Network']['Mole_layers_unique']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer_CDKN.append(copy.deepcopy(layer))
        self.encoder_norm_share = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_IDH = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_pq = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_CDKN = LayerNorm(self.size[1], eps=1e-6)

    def forward(self, hidden_states):
        for layer_block in self.layer_share:
            hidden_states, weights = layer_block(hidden_states)
        hidden_states_share = self.encoder_norm_share(hidden_states)  # [B,2500,512]

        count_IDH=0
        for layer_block in self.layer_IDH:
            if count_IDH==0:
                hidden_states_IDH, weights = layer_block(hidden_states_share)
                count_IDH+=1
            else:
                hidden_states_IDH, weights = layer_block(hidden_states_IDH)
        hidden_states_IDH = self.encoder_norm_IDH(hidden_states_IDH)  # [B,2500,512]

        count_pq = 0
        for layer_block in self.layer_pq:
            if count_pq == 0:
                hidden_states_pq, weights = layer_block(hidden_states_share)
                count_pq += 1
            else:
                hidden_states_pq, weights = layer_block(hidden_states_pq)
        hidden_states_pq = self.encoder_norm_pq(hidden_states_pq)  # [B,2500,512]

        count_CDKN= 0
        for layer_block in self.layer_CDKN:
            if count_CDKN == 0:
                hidden_states_CDKN, weights = layer_block(hidden_states_share)
                count_CDKN += 1
            else:
                hidden_states_CDKN, weights = layer_block(hidden_states_CDKN)
        hidden_states_CDKN = self.encoder_norm_CDKN(hidden_states_CDKN)  # [B,2500,512]

        return hidden_states_IDH,hidden_states_pq,hidden_states_CDKN


class Mine_molecular_predall(nn.Module):
    def __init__(self, opt):
        super(Mine_molecular_predall, self).__init__()
        self.opt = opt
        self.size = [1024, 512]
        self.layer_share = nn.ModuleList()
        for _ in range(self.opt['Network']['Mole_layers_share']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer_share.append(copy.deepcopy(layer))

        self.layer = nn.ModuleList()
        for _ in range(self.opt['Network']['Mole_layers_unique']):
            layer = Block(opt, self.size[1], vis=False)
            self.layer.append(copy.deepcopy(layer))

        self.encoder_norm_share = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm = LayerNorm(self.size[1], eps=1e-6)


    def forward(self, hidden_states):
        for layer_block in self.layer_share:
            hidden_states, weights = layer_block(hidden_states)
        hidden_states_share = self.encoder_norm_share(hidden_states)  # [B,2500,512]

        count=0
        for layer_block in self.layer:
            if count==0:
                hidden_states, weights = layer_block(hidden_states_share)
                count+=1
            else:
                hidden_states, weights = layer_block(hidden_states)
        hidden_states = self.encoder_norm(hidden_states)  # [B,2500,512]
        return [hidden_states]

class surv_pred(nn.Module):
    def __init__(self):
        super(surv_pred, self).__init__()
        self._fc_surv = nn.Linear(59, 2)


    def forward(self,encoded_his,encoded_gene,age,sex,IDH,pq,CDKN,subtype,grade):

        age=torch.unsqueeze(age,1)
        sex = torch.unsqueeze(sex, 1)
        IDH = torch.unsqueeze(IDH, 1)
        pq = torch.unsqueeze(pq, 1)
        CDKN = torch.unsqueeze(CDKN, 1)
        subtype = torch.unsqueeze(subtype, 1)
        grade = torch.unsqueeze(grade, 1)

        encoded_feature = torch.cat((encoded_his,encoded_gene),dim=1)
        encoded_feature= torch.cat((encoded_feature,age,sex,IDH,pq,CDKN,subtype,grade),dim=1)

        logits = self._fc_surv(encoded_feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, F.softmax(logits)


class Label_correlation_Graph(nn.Module):
    def __init__(self, opt):
        super(Label_correlation_Graph, self).__init__()
        self.opt = opt

        self.size = [1024, 512]

        self.alpha=self.opt['Network']['graph_alpha']
        self.n_classes_IDH=2
        self.n_classes_CDKN=2
        self.n_classes_1p19q=2

        if self.opt['TrainingSet']=='All':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1.76])).float())
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 5.5])).float())
            self.criterion_ce_CDKN = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.26, 1])).float())
            self.adj = np.array([[1, 0.37, 0.31], [0.95, 1, 0.13], [0.23, 0.035, 1]])
        elif self.opt['TrainingSet']=='TCGA':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1])).float())
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 4.4])).float())
            self.criterion_ce_CDKN = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.5, 1])).float())
            self.adj = np.array([[1,0.4038,0.3035],[1,1,0.1263],[0.2595,0.0436,1]])

        self.encoder_norm_IDH = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_1p19q = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_CDKN = LayerNorm(self.size[1], eps=1e-6)

        self.gc1 = GraphConvolution(self.size[1], self.size[1],self.adj)
        self.gc2 = GraphConvolution(self.size[1], 2,self.adj)
        self.dropout=Dropout(0.5)

        # atten
        self.attention_IDH = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_1p19q = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_CDKN = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_V_IDH = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_IDH = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_IDH = nn.Linear(128, 1)
        self.attention_V_1p19q = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_1p19q = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_1p19q = nn.Linear(128, 1)
        self.attention_V_CDKN = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_CDKN = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_CDKN = nn.Linear(128, 1)

        self._fc2_IDH_1 = nn.Linear(self.size[1], self.n_classes_IDH)
        self._fc2_CDKN_1 = nn.Linear(self.size[1], self.n_classes_CDKN)
        self._fc2_1p19q_1 = nn.Linear(self.size[1], self.n_classes_1p19q)


    def forward(self, encoded_IDH,encoded_1p19q,encoded_CDKN):


        encoded_IDH=torch.unsqueeze(encoded_IDH, dim=3) #[BS,2500,512,1]
        encoded_1p19q = torch.unsqueeze(encoded_1p19q, dim=3)
        encoded_CDKN = torch.unsqueeze(encoded_CDKN, dim=3)
        GCN_input=torch.cat((encoded_IDH,encoded_1p19q,encoded_CDKN),3)#[BS,2500,512,3 ]
        GCN_output =  F.relu(self.gc1(GCN_input))#[BS,2500,512,3 ]
        GCN_output=GCN_output*self.alpha+GCN_input*(1-self.alpha)#[BS,2500,512,3 ]

        ########################   IDH   ########################
        encoded_IDH=GCN_output[...,0]#[BS,2500,512]
        encoded_IDH = self.encoder_norm_IDH(encoded_IDH)#[BS,2500,512]
        encoded_IDH_ori = encoded_IDH
        A_V_IDH = self.attention_V_IDH(encoded_IDH)  # BxNx128
        A_U_IDH = self.attention_U_IDH(encoded_IDH)  # BxNx128
        A_encoded_IDH = self.attention_weights_IDH(A_V_IDH * A_U_IDH)  # BxNx1
        A_encoded_IDH = F.softmax(A_encoded_IDH, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_IDH.shape[0]):
            if i == 0:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_IDH_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.cat((Final_con_layer_IDH, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_IDH_new = torch.cat(
                    (encoded_IDH_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_IDH = encoded_IDH_new  # Bx512

        ########################   1p19q   ########################
        encoded_1p19q = GCN_output[..., 1]  # [BS,2500,512]
        encoded_1p19q = self.encoder_norm_1p19q(encoded_1p19q)  # [BS,2500,512]
        encoded_1p19q_ori = encoded_1p19q
        A_V_1p19q = self.attention_V_1p19q(encoded_1p19q)  # BxNx128
        A_U_1p19q = self.attention_U_1p19q(encoded_1p19q)  # BxNx128
        A_encoded_1p19q = self.attention_weights_1p19q(A_V_1p19q * A_U_1p19q)  # BxNx1
        A_encoded_1p19q = F.softmax(A_encoded_1p19q, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_1p19q.shape[0]):
            if i == 0:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_1p19q_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.cat((Final_con_layer_1p19q, torch.unsqueeze(Final_con_layer, 0)),
                                                  dim=0)  # BSxNx512
                encoded_1p19q_new = torch.cat(
                    (encoded_1p19q_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_1p19q = encoded_1p19q_new  # Bx512

        ########################   CDKN   ########################
        encoded_CDKN = GCN_output[..., 2]#[BS,2500,512]
        encoded_CDKN = self.encoder_norm_CDKN(encoded_CDKN)#[BS,2500,512]
        encoded_CDKN_ori = encoded_CDKN
        A_V_CDKN = self.attention_V_CDKN(encoded_CDKN)  # BxNx128
        A_U_CDKN = self.attention_U_CDKN(encoded_CDKN)  # BxNx128
        A_encoded_CDKN = self.attention_weights_CDKN(A_V_CDKN * A_U_CDKN)  # BxNx1
        A_encoded_CDKN = F.softmax(A_encoded_CDKN, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_CDKN.shape[0]):
            if i == 0:
                Final_con_layer = encoded_CDKN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_CDKN = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_CDKN_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_CDKN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_CDKN = torch.cat((Final_con_layer_CDKN, torch.unsqueeze(Final_con_layer, 0)),dim=0)  # BSxNx512
                encoded_CDKN_new = torch.cat(
                    (encoded_CDKN_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_CDKN = encoded_CDKN_new  # Bx512

        ####################saliency maps for IDH wt
        weight_IDH_wt = torch.unsqueeze(self._fc2_IDH_1.weight[0], dim=1)  # [512,1]

        weight_IDH_wt_temp = np.array(weight_IDH_wt.tolist())[...,0]
        index = np.argsort(np.abs(weight_IDH_wt_temp))
        # weight_IDH_wt=weight_IDH_wt[list(index[int(512*0.99):]),:]
        # Final_con_layer_IDH0=Final_con_layer_IDH[...,list(index[int(512*0.99):])]

        saliency_IDH_wt = torch.matmul(Final_con_layer_IDH, weight_IDH_wt)[..., 0]  # [BSxN]
        if self._fc2_IDH_1.bias is not None:
            saliency_IDH_wt = saliency_IDH_wt + self._fc2_IDH_1.bias[0] / encoded_IDH_ori.shape[1]  # [BSxN]


        ####################saliency maps for IDH mut
        weight_IDH_mut = torch.unsqueeze(self._fc2_IDH_1.weight[1], dim=1)  # [512,1]
        saliency_IDH_mut = torch.matmul(Final_con_layer_IDH, weight_IDH_mut)[..., 0]  # [BSxN]
        if self._fc2_IDH_1.bias is not None:
            saliency_IDH_mut = saliency_IDH_mut + self._fc2_IDH_1.bias[1] / encoded_IDH_ori.shape[1]  # [BSxN]

        ####################saliency maps for 1p19q codel
        weight_1p19q_codel = torch.unsqueeze(self._fc2_1p19q_1.weight[1], dim=1)  # [512,1]
        saliency_1p19q_codel = torch.matmul(Final_con_layer_1p19q, weight_1p19q_codel)[..., 0]  # [BSxN]
        if self._fc2_1p19q_1.bias is not None:
            saliency_1p19q_codel = saliency_1p19q_codel + self._fc2_1p19q_1.bias[1] / encoded_1p19q_ori.shape[1]  # [BSxN]

        ####################saliency maps for 1p19q noncodel
        weight_1p19q_noncodel = torch.unsqueeze(self._fc2_1p19q_1.weight[0], dim=1)  # [512,1]
        saliency_1p19q_noncodel = torch.matmul(Final_con_layer_1p19q, weight_1p19q_noncodel)[..., 0]  # [BSxN]
        if self._fc2_1p19q_1.bias is not None:
            saliency_1p19q_noncodel = saliency_1p19q_noncodel + self._fc2_1p19q_1.bias[0] / encoded_1p19q_ori.shape[
                1]  # [BSxN]

        ####################saliency maps for CDKN HOMDEL
        weight_CDKN_HOMDEL = torch.unsqueeze(self._fc2_CDKN_1.weight[1], dim=1)  # [512,1]
        saliency_CDKN_HOMDEL = torch.matmul(Final_con_layer_CDKN, weight_CDKN_HOMDEL)[..., 0]  # [BSxN]
        if self._fc2_CDKN_1.bias is not None:
            saliency_CDKN_HOMDEL = saliency_CDKN_HOMDEL + self._fc2_CDKN_1.bias[1] / encoded_CDKN_ori.shape[
                1]  # [BSxN]

        ####################saliency maps for CDKN NonHOMDEL
        weight_CDKN_NonHOMDEL = torch.unsqueeze(self._fc2_CDKN_1.weight[0], dim=1)  # [512,1]
        saliency_CDKN_NonHOMDEL = torch.matmul(Final_con_layer_CDKN, weight_CDKN_NonHOMDEL)[..., 0]  # [BSxN]
        if self._fc2_CDKN_1.bias is not None:
            saliency_CDKN_NonHOMDEL = saliency_CDKN_NonHOMDEL + self._fc2_CDKN_1.bias[0] / encoded_CDKN_ori.shape[1]  # [BSxN]

        logits_CDKN = self._fc2_CDKN_1(encoded_CDKN)#[BS,2]
        logits_IDH = self._fc2_IDH_1(encoded_IDH)  # [BS,2]
        logits_1p19q = self._fc2_1p19q_1(encoded_1p19q)#[BS,2]

        results_dict = {'logits_IDH': logits_IDH,'logits_1p19q': logits_1p19q,'logits_CDKN': logits_CDKN}



        # return results_dict,saliency_IDH_wt,saliency_IDH_mut,saliency_1p19q_codel,saliency_1p19q_noncodel,\
        #        saliency_CDKN_HOMDEL,saliency_CDKN_NonHOMDEL,encoded_IDH,encoded_1p19q,encoded_CDKN,encoded_IDH[:, list(index[int(512 * 0.95):])]
        return results_dict,saliency_IDH_wt,saliency_1p19q_codel,encoded_IDH,encoded_1p19q,encoded_CDKN



    def calculateLoss_Graph(self,encoded_IDH,encoded_1p19q,encoded_CDKN):

        dis_IDH_IDH = F.cosine_similarity(encoded_IDH, encoded_IDH, dim=1)
        dis_IDH_1p19 = F.cosine_similarity(encoded_IDH, encoded_1p19q, dim=1)
        dis_IDH_CDKN = F.cosine_similarity(encoded_IDH, encoded_CDKN, dim=1)
        dis_1p19_IDH = F.cosine_similarity(encoded_1p19q, encoded_IDH, dim=1)
        dis_1p19_1p19 = F.cosine_similarity(encoded_1p19q, encoded_1p19q, dim=1)
        dis_1p19_CDKN = F.cosine_similarity(encoded_1p19q, encoded_CDKN, dim=1)
        dis_CDKN_IDH = F.cosine_similarity(encoded_CDKN, encoded_IDH, dim=1)
        dis_CDKN_1p19 = F.cosine_similarity(encoded_CDKN, encoded_1p19q, dim=1)
        dis_CDKN_CDKN = F.cosine_similarity(encoded_CDKN, encoded_CDKN, dim=1)

        cos_dis_matrix=[dis_IDH_IDH,dis_IDH_1p19,dis_IDH_CDKN,dis_1p19_IDH,dis_1p19_1p19,dis_1p19_CDKN,dis_CDKN_IDH,
                        dis_CDKN_1p19,dis_CDKN_CDKN]

        adj_T = self.adj.T
        adj = (adj_T + self.adj) / 2
        adj=torch.from_numpy(np.array(adj)).float().cuda(self.opt['gpus'][0])
        adj=torch.unsqueeze(adj,dim=0)
        adj =adj.repeat(dis_IDH_IDH.detach().cpu().numpy().shape[0],1,1)


        dis_1p19_CDKN = dis_1p19_CDKN.detach().cpu().numpy()  # [BS]
        dis_1p19_CDKN_FLAG=np.ones(dis_IDH_IDH.detach().cpu().numpy().shape[0],dtype=float)
        for i in range(dis_IDH_IDH.detach().cpu().numpy().shape[0]):
            if dis_1p19_CDKN[i]<0.1:
                dis_1p19_CDKN_FLAG[i]=0
        dis_1p19_CDKN_FLAG=torch.from_numpy(np.array(dis_1p19_CDKN_FLAG)).float().cuda(self.opt['gpus'][0])

        dis_CDKN_1p19 = dis_CDKN_1p19.detach().cpu().numpy()  # [BS]
        dis_CDKN_1p19_FLAG = np.ones(dis_IDH_IDH.detach().cpu().numpy().shape[0], dtype=float)
        for i in range(dis_IDH_IDH.detach().cpu().numpy().shape[0]):
            if dis_CDKN_1p19[i] < 0.1:
                dis_CDKN_1p19_FLAG[i] = 0
        dis_CDKN_1p19_FLAG = torch.from_numpy(np.array(dis_CDKN_1p19_FLAG)).float().cuda(self.opt['gpus'][0])


        self.loss_Graph = (cos_dis_matrix[0] - adj[:, 0, 0]) ** 2 + (cos_dis_matrix[1] - adj[:, 0, 1]) ** 2 + (cos_dis_matrix[2] - adj[:, 0, 2]) ** 2 \
                          + (cos_dis_matrix[3] - adj[:, 1, 0]) ** 2 + (cos_dis_matrix[4] - adj[:, 1, 1]) ** 2 +dis_1p19_CDKN_FLAG*(cos_dis_matrix[5] - adj[:, 1, 2]) ** 2 \
                          + (cos_dis_matrix[6] - adj[:, 2, 0]) ** 2 + dis_CDKN_1p19_FLAG*(cos_dis_matrix[7] - adj[:, 2, 1]) ** 2 + (cos_dis_matrix[8] - adj[:, 2, 2]) ** 2
        return torch.mean(self.loss_Graph)



    def calculateLoss_IDH(self, pred, label):
        FLAT_normal = False
        self.loss_IDH = 0
        count=0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count==0:
                    pred_IDH = pred[i].unsqueeze(0)
                    label_IDH = label[i].unsqueeze(0)
                    count+=1
                else:
                    pred_IDH = torch.cat((pred_IDH, pred[i].unsqueeze(0)), 0)
                    label_IDH = torch.cat((label_IDH, label[i].unsqueeze(0)), 0)
                FLAT_normal= True
            else:
                continue

        if not FLAT_normal:
            self.loss_IDH=0
        else:
            self.loss_IDH = self.criterion_ce_IDH(pred_IDH, label_IDH)
        return self.loss_IDH

    def calculateLoss_1p19q(self, pred, label):
        FLAT_normal = False
        self.loss_1p19q = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_1p19q = pred[i].unsqueeze(0)
                    label_1p19q = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_1p19q = torch.cat((pred_1p19q, pred[i].unsqueeze(0)), 0)
                    label_1p19q = torch.cat((label_1p19q, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_1p19q = 0
        else:
            self.loss_1p19q = self.criterion_ce_1p19q(pred_1p19q, label_1p19q)
        return self.loss_1p19q

    def calculateLoss_CDKN(self, pred, label):
        FLAT_normal = False
        self.loss_CDKN = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_CDKN = pred[i].unsqueeze(0)
                    label_CDKN = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_CDKN = torch.cat((pred_CDKN, pred[i].unsqueeze(0)), 0)
                    label_CDKN = torch.cat((label_CDKN, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_CDKN = 0
        else:
            self.loss_CDKN = self.criterion_ce_CDKN(pred_CDKN, label_CDKN)
        return self.loss_CDKN



class Label_correlation_predall(nn.Module):
    def __init__(self, opt):
        super(Label_correlation_predall, self).__init__()
        self.opt = opt
        self.surv= True if self.opt['name'].split('_')[0]=='surv' else False
        if self.surv:
            self.mode=self.opt['name'].split('_')[1]
        self.size = [1024, 512]

        self.n_classes=2


        if self.opt['marker']=='MGMT':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == '710':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'ATRX':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'EGFR':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.7,1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'TERT':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'OLIG2':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([45,1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'PDGFRA':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 10])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'PTEN':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 3.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        if self.opt['marker'] == 'P53':
            self.criterion_ce= nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])

        self.encoder_norm = LayerNorm(self.size[1], eps=1e-6)

        # atten

        self.attention = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self._fc2_1 = nn.Linear(self.size[1], self.n_classes)



    def forward(self, encoded):
        hidden_states=encoded[0]
        encoded = hidden_states  # [BS,2500,512]
        A_V = self.attention_V(encoded)  # BxNx128
        A_U = self.attention_U(encoded)  # BxNx128
        A_encoded = self.attention_weights(A_V * A_U)  # BxNx1
        A_encoded = F.softmax(A_encoded, dim=1)[..., 0]  # BxN AMIL attention map
        encoded_IDH=encoded
        for i in range(encoded_IDH.shape[0]):
            if i == 0:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_IDH_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.cat((Final_con_layer_IDH, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_IDH_new = torch.cat(
                    (encoded_IDH_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded = encoded_IDH_new  # Bx512


        logits = self._fc2_1(encoded)  # [BS,2]

        results_dict = {'logits': logits}
        return results_dict


    def calculateLoss(self, pred, label):
        FLAT_normal = False
        self.loss_IDH = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_IDH = pred[i].unsqueeze(0)
                    label_IDH = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_IDH = torch.cat((pred_IDH, pred[i].unsqueeze(0)), 0)
                    label_IDH = torch.cat((label_IDH, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_IDH = 0
        else:
            self.loss_IDH = self.criterion_ce(pred_IDH, label_IDH)
        return self.loss_IDH


class Mine_His(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_His, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]

        self.layer_His = nn.ModuleList()
        for _ in range(self.opt['Network']['His_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_His.append(copy.deepcopy(layer))
        self.encoder_norm_His = LayerNorm(self.size[1], eps=1e-6)

        self.layer_Grade = nn.ModuleList()
        for _ in range(self.opt['Network']['His_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_Grade.append(copy.deepcopy(layer))
        self.encoder_norm_Grade = LayerNorm(self.size[1], eps=1e-6)
        if self.opt['TrainingSet']=='All':
            self.criterion_ce_His = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4.2,5.3, 1])).float())
            self.criterion_ce_Grade = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([3.2, 3.6, 1])).float())
            self.criterion_ce_His_noise = nn.CrossEntropyLoss()
            self.criterion_ce_Grade_noise = nn.CrossEntropyLoss()
        elif self.opt['TrainingSet']=='TCGA':
            self.criterion_ce_His = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([3, 2.5, 1])).float())
            # self.criterion_ce_His = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 2.2, 3.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_Grade = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2, 2, 1])).float())
            self.criterion_ce_His_noise = nn.CrossEntropyLoss()
            self.criterion_ce_Grade_noise = nn.CrossEntropyLoss()

    def forward(self, hidden_states,imp_his, imp_grade):
        """
            hidden_states: (B,N/N+1,512)
            imp_his: torch (B,N/N+1)
            imp_grade: torch (B,N/N+1)
        """
        attn_weights_His = []
        attn_weights_Grade = []

        imp_his = imp_his.unsqueeze(2).repeat(1, 1, self.size[1])
        imp_grade = imp_grade.unsqueeze(2).repeat(1, 1, self.size[1])
        hidden_states_his=hidden_states*(1-self.opt['Network']['atten_theta'])*imp_his+hidden_states*self.opt['Network']['atten_theta']
        hidden_states_grade = hidden_states*(1-self.opt['Network']['atten_theta'])*imp_grade+hidden_states*self.opt['Network']['atten_theta']

        for layer_block in self.layer_His:
            hidden_states_his, weights = layer_block(hidden_states_his)
            if self.vis:
                attn_weights_His.append(weights)
        encoded_His = self.encoder_norm_His(hidden_states_his)  # [B,2500,512]

        for layer_block in self.layer_Grade:
            hidden_states_grade, weights = layer_block(hidden_states_grade)
            if self.vis:
                attn_weights_Grade.append(weights)
        encoded_Grade = self.encoder_norm_Grade(hidden_states_grade)  # [B,2500,512]

        return hidden_states_his, hidden_states_grade,encoded_His,encoded_Grade

    def calculateLoss_His_ori(self, pred, label):
        label_his = label.detach().cpu().numpy()
        FLAT_OA = False
        FLAT_normal = False
        self.loss_His_OA = 0
        self.loss_His_normal = 0
        for i in range(label_his.shape[0]):
            if label_his[i] == 3:
                if not FLAT_OA:
                    FLAT_OA = True
                    pred_OA = pred[i].unsqueeze(0)
                    label_OA = label[i].unsqueeze(0)
                else:
                    pred_OA = torch.cat((pred_OA, pred[i].unsqueeze(0)), 0)
                    label_OA = torch.cat((label_OA, label[i].unsqueeze(0)), 0)
            else:
                if not FLAT_normal:
                    FLAT_normal = True
                    pred_normal = pred[i].unsqueeze(0)
                    label_normal = label[i].unsqueeze(0)
                else:
                    pred_normal = torch.cat((pred_normal, pred[i].unsqueeze(0)), 0)
                    label_normal = torch.cat((label_normal, label[i].unsqueeze(0)), 0)

        if FLAT_normal:
            self.loss_His_normal = self.criterion_ce_His(pred_normal, label_normal)

        if FLAT_OA:
            for i in range(label_OA.detach().cpu().numpy().shape[0]):
                pred_OA = F.softmax(pred_OA, dim=1)
                self.loss_His_OA += (-1 * torch.log(1 - pred_OA[i, 2]))
            self.loss_His_OA = self.loss_His_OA / (label_OA.detach().cpu().numpy().shape[0])

        self.loss_His = (self.loss_His_normal * (
            label_normal.detach().cpu().numpy().shape[0] if self.loss_His_normal else 0) +
                         self.loss_His_OA * self.opt['Network']['w_loss_OA'] * (
                             label_OA.detach().cpu().numpy().shape[0] if self.loss_His_OA else 0)
                         ) /label.detach().cpu().numpy().shape[0]

        return self.loss_His_normal+self.loss_His_OA*self.opt['Network']['w_loss_OA']

    def calculateLoss_His(self, pred, label,file_name,His_Train_noise_list):
        label_his=label.detach().cpu().numpy()
        FLAT_OA=False
        FLAT_normal = False
        FLAT_noise = False
        self.loss_His_OA=0
        self.loss_His_normal=0
        self.loss_His_noise = 0
        for i in range(label_his.shape[0]):
            if label_his[i]==3:
                if not FLAT_OA:
                    FLAT_OA=True
                    pred_OA =pred[i].unsqueeze(0)
                    label_OA=label[i].unsqueeze(0)
                else:
                    pred_OA =torch.cat((pred_OA, pred[i].unsqueeze(0)), 0)
                    label_OA = torch.cat((label_OA, label[i].unsqueeze(0)), 0)
            elif file_name[i] in His_Train_noise_list:  #
                if not FLAT_noise:
                    FLAT_noise = True
                    pred_noise = pred[i].unsqueeze(0)
                    label_noise = label[i].unsqueeze(0)
                else:
                    pred_noise = torch.cat((pred_noise, pred[i].unsqueeze(0)), 0)
                    label_noise = torch.cat((label_noise, label[i].unsqueeze(0)), 0)
            else:
                if not FLAT_normal:
                    FLAT_normal = True
                    pred_normal = pred[i].unsqueeze(0)
                    label_normal = label[i].unsqueeze(0)
                else:
                    pred_normal = torch.cat((pred_normal, pred[i].unsqueeze(0)), 0)
                    label_normal = torch.cat((label_normal, label[i].unsqueeze(0)), 0)


        if FLAT_normal:
            self.loss_His_normal = self.criterion_ce_His(pred_normal, label_normal)


        if FLAT_OA:
            for i in range(label_OA.detach().cpu().numpy().shape[0]):
                pred_OA=F.softmax(pred_OA,dim=1)
                self.loss_His_OA+=(-1*torch.log(1-pred_OA[i,2]))
            self.loss_His_OA=self.loss_His_OA/(label_OA.detach().cpu().numpy().shape[0])

        if FLAT_noise:
            pred_sm = F.softmax(pred_noise)
            for i in range(label_noise.detach().cpu().numpy().shape[0]):

                pred_His_ori = pred_sm.detach().cpu().numpy()[i]
                entropy_temp = 0
                for j in range(pred_His_ori.shape[0]):
                    entropy_temp += pred_His_ori[j] * math.log2(pred_His_ori[j]) * (-1)
                if entropy_temp < self.opt['Network']['entropy_thre']:
                    _, label_net = torch.max(pred_noise[i].unsqueeze(0).data, 1)
                    self.loss_His_noise += self.criterion_ce_His_noise(pred_noise[i].unsqueeze(0), label_net)
                else:
                    _, label_net = torch.max(pred_noise[i].unsqueeze(0).data, 1)
                    self.loss_His_noise += (self.criterion_ce_His_noise(pred_noise[i].unsqueeze(0), label_noise[i].unsqueeze(0)) + self.criterion_ce_His_noise(pred_noise[i].unsqueeze(0), label_net)) / 2
            self.loss_His_noise=self.loss_His_noise/(label_noise.detach().cpu().numpy().shape[0])

        self.loss_His = (self.loss_His_normal*(label_normal.detach().cpu().numpy().shape[0] if self.loss_His_normal else 0)+
                         self.loss_His_OA*self.opt['Network']['w_loss_OA']*(label_OA.detach().cpu().numpy().shape[0] if self.loss_His_OA else 0)+
                         self.loss_His_noise*self.opt['Network']['w_loss_noise']*(label_noise.detach().cpu().numpy().shape[0] if self.loss_His_noise else 0))/label.detach().cpu().numpy().shape[0]

        return self.loss_His


    def calculateLoss_Grade_ori(self, pred, label):

        self.loss_Grade = self.criterion_ce_Grade(pred, label)
        return self.loss_Grade
    def calculateLoss_Grade(self, pred, label,file_name,Grade_Train_noise_list):
        label_grade = label.detach().cpu().numpy()
        FLAT_normal = False
        FLAT_noise = False
        self.loss_Grade_normal = 0
        self.loss_Grade_noise = 0
        for i in range(label_grade.shape[0]):
            if file_name[i] in Grade_Train_noise_list:  #
                if not FLAT_noise:
                    FLAT_noise = True
                    pred_noise = pred[i].unsqueeze(0)
                    label_noise = label[i].unsqueeze(0)
                else:
                    pred_noise = torch.cat((pred_noise, pred[i].unsqueeze(0)), 0)
                    label_noise = torch.cat((label_noise, label[i].unsqueeze(0)), 0)
            else:
                if not FLAT_normal:
                    FLAT_normal = True
                    pred_normal = pred[i].unsqueeze(0)
                    label_normal = label[i].unsqueeze(0)
                else:
                    pred_normal = torch.cat((pred_normal, pred[i].unsqueeze(0)), 0)
                    label_normal = torch.cat((label_normal, label[i].unsqueeze(0)), 0)

        if FLAT_normal:
            self.loss_Grade_normal = self.criterion_ce_Grade(pred_normal, label_normal)

        if FLAT_noise:
            pred_sm = F.softmax(pred_noise)
            for i in range(label_noise.detach().cpu().numpy().shape[0]):
                pred_Grade_ori = pred_sm.detach().cpu().numpy()[i]
                entropy_temp = 0
                for j in range(pred_Grade_ori.shape[0]):
                    entropy_temp += pred_Grade_ori[j] * math.log2(pred_Grade_ori[j]) * (-1)
                if entropy_temp < self.opt['Network']['entropy_thre']:
                    _, label_net = torch.max(pred_noise[i].unsqueeze(0).data, 1)
                    self.loss_Grade_noise += self.criterion_ce_Grade_noise(pred_noise[i].unsqueeze(0), label_net)
                else:
                    _, label_net = torch.max(pred_noise[i].unsqueeze(0).data, 1)
                    self.loss_Grade_noise += (self.criterion_ce_Grade_noise(pred_noise[i].unsqueeze(0),label_noise[i].unsqueeze(0)) + self.criterion_ce_Grade_noise(
                        pred_noise[i].unsqueeze(0), label_net)) / 2
            self.loss_Grade_noise = self.loss_Grade_noise / (label_noise.detach().cpu().numpy().shape[0])

        self.loss_Grade = (self.loss_Grade_normal * (
            label_normal.detach().cpu().numpy().shape[0] if self.loss_Grade_normal else 0) +
                           self.loss_Grade_noise *self.opt['Network']['w_loss_noise']* (label_noise.detach().cpu().numpy().shape[0] if self.loss_Grade_noise else 0)) / label.detach().cpu().numpy().shape[0]

        return self.loss_Grade
        # self.loss_Grade = self.criterion_ce_Grade(pred, label)
        # return self.loss_Grade



class Cls_Diag_endtoend(nn.Module):
    def __init__(self, opt):
        super(Cls_Diag_endtoend, self).__init__()
        self.opt = opt
        self.Name=self.opt['Clstype']
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)



        if self.Name=='Diag':
            self.n_classes= 6
            self.fc = nn.Linear(512, self.n_classes)
            if self.opt['TrainingSet']=='TCGA':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([0.4,4,9,6,7,5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            elif self.opt['TrainingSet']=='All':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([0.8, 6.7, 19, 14, 11, 9])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        #[1, 7, 16, 12, 10, 9]
        elif self.Name=='DiagSim':
            self.n_classes = 4
            self.fc = nn.Linear(512, self.n_classes)
            if self.opt['TrainingSet'] == 'TCGA':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([1, 4, 3.5,2.8])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            elif self.opt['TrainingSet'] == 'All':
                self.criterion_ce = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(np.array([1,6.8,6.8,4.8])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
    def forward(self, encoded):
        """
            encoded: [B, N,512]
        """
        A_V = self.attention_V(encoded)  # BxNx128
        A_U = self.attention_U(encoded)  # BxNx128
        A_encoded = self.attention_weights(A_V * A_U)  # BxNx1
        A_encoded = F.softmax(A_encoded, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded.shape[0]):
            if i == 0:
                Final_con_layer = encoded[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, encoded[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_His_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, encoded[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.cat((Final_con_layer_His, torch.unsqueeze(Final_con_layer, 0)),dim=0)  # BSxNx512
                encoded_His_new = torch.cat((encoded_His_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded = encoded_His_new  # Bx512

        ####################saliency maps for G4GBM
        weight_G4GBM = torch.unsqueeze(self.fc.weight[0], dim=1)  # [512,1]
        saliency_G4GBM = torch.matmul(Final_con_layer_His, weight_G4GBM)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G4GBM = saliency_G4GBM + self.fc.bias[0] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G4A
        weight_G4A = torch.unsqueeze(self.fc.weight[1], dim=1)  # [512,1]
        saliency_G4A = torch.matmul(Final_con_layer_His, weight_G4A)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G4A = saliency_G4A + self.fc.bias[1] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G3A
        weight_G3A = torch.unsqueeze(self.fc.weight[2], dim=1)  # [512,1]
        saliency_G3A = torch.matmul(Final_con_layer_His, weight_G3A)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G3A = saliency_G3A + self.fc.bias[2] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G2A
        weight_G2A = torch.unsqueeze(self.fc.weight[3], dim=1)  # [512,1]
        saliency_G2A = torch.matmul(Final_con_layer_His, weight_G2A)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G2A = saliency_G2A + self.fc.bias[3] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G3O
        weight_G3O = torch.unsqueeze(self.fc.weight[4], dim=1)  # [512,1]
        saliency_G3O = torch.matmul(Final_con_layer_His, weight_G3O)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G3O = saliency_G3O + self.fc.bias[4] / encoded.shape[1]  # [BSxN]

        ####################saliency maps for G2O
        weight_G2O = torch.unsqueeze(self.fc.weight[5], dim=1)  # [512,1]
        saliency_G2O = torch.matmul(Final_con_layer_His, weight_G2O)[..., 0]  # [BSxN]
        if self.fc.bias is not None:
            saliency_G2O = saliency_G2O + self.fc.bias[5] / encoded.shape[1]  # [BSxN

        logits = self.fc(encoded)  # [BS,cls]
        results_dict = {'logits': logits}
        # return results_dict,saliency_G4GBM,saliency_G4A,saliency_G3A,saliency_G2A,saliency_G3O,saliency_G2O
        return results_dict
        # ####################saliency maps for G4GBM
        # weight_G4GBM = torch.unsqueeze(self.fc.weight[0], dim=1)  # [512,1]
        # saliency_G4GBM = torch.matmul(Final_con_layer_His, weight_G4GBM)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G4GBM = saliency_G4GBM + self.fc.bias[0] / encoded.shape[1]  # [BSxN]
        #
        # ####################saliency maps for G4A
        # weight_G4A = torch.unsqueeze(self.fc.weight[1], dim=1)  # [512,1]
        # saliency_G4A = torch.matmul(Final_con_layer_His, weight_G4A)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G4A = saliency_G4A + self.fc.bias[1] / encoded.shape[1]  # [BSxN]
        #
        # ####################saliency maps for G23A
        # weight_G3A = torch.unsqueeze(self.fc.weight[2], dim=1)  # [512,1]
        # saliency_G23A = torch.matmul(Final_con_layer_His, weight_G3A)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G23A = saliency_G23A + self.fc.bias[2] / encoded.shape[1]  # [BSxN]
        #
        # ####################saliency maps for G23O
        # weight_G2A = torch.unsqueeze(self.fc.weight[3], dim=1)  # [512,1]
        # saliency_G23O = torch.matmul(Final_con_layer_His, weight_G2A)[..., 0]  # [BSxN]
        # if self.fc.bias is not None:
        #     saliency_G23O = saliency_G23O + self.fc.bias[3] / encoded.shape[1]  # [BSxN]
        #
        #
        # logits = self.fc(encoded)  # [BS,cls]
        # results_dict = {'logits': logits}
        # return results_dict

    def calculateLoss(self, pred0, GT):
        FLAT_normal = False
        self.loss = 0
        count = 0
        NA_cls= 4 if self.Name=='DiagSim' else 6
        for i in range(GT.detach().cpu().numpy().shape[0]):
            if GT.detach().cpu().numpy()[i] != NA_cls:
                if count == 0:
                    pred = pred0[i].unsqueeze(0)
                    label = GT[i].unsqueeze(0)
                    count += 1
                else:
                    pred = torch.cat((pred, pred0[i].unsqueeze(0)), 0)
                    label = torch.cat((label, GT[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss = 0
        else:
            self.loss = self.criterion_ce(pred, label)
        return self.loss




class Cls_Diag_endtoend_marker(nn.Module):
    def __init__(self, opt):
        super(Cls_Diag_endtoend_marker, self).__init__()
        self.opt = opt
        self.marker = self.opt['name'].split('_')[2]

        self.n_classes = 2
        self.fc = nn.Linear(512, self.n_classes)
        if self.marker == 'IDH':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 6.7])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.marker == '1p19q':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 4])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.marker == 'CDKN':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])


        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)


    def forward(self, encoded):
        """
            encoded: [B, N,512]
        """
        A_V = self.attention_V(encoded)  # BxNx128
        A_U = self.attention_U(encoded)  # BxNx128
        A_encoded = self.attention_weights(A_V * A_U)  # BxNx1
        A_encoded = F.softmax(A_encoded, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded.shape[0]):
            if i == 0:
                Final_con_layer = encoded[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, encoded[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_His_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded[i], 1).expand(-1, encoded[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.cat((Final_con_layer_His, torch.unsqueeze(Final_con_layer, 0)),dim=0)  # BSxNx512
                encoded_His_new = torch.cat((encoded_His_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded = encoded_His_new  # Bx512

        logits = self.fc(encoded)  # [BS,cls]
        results_dict = {'logits': logits}
        return results_dict


    def calculateLoss(self, pred0, GT):
        FLAT_normal = False
        self.loss = 0
        count = 0
        NA_cls= 2
        for i in range(GT.detach().cpu().numpy().shape[0]):
            if GT.detach().cpu().numpy()[i] != NA_cls:
                if count == 0:
                    pred = pred0[i].unsqueeze(0)
                    label = GT[i].unsqueeze(0)
                    count += 1
                else:
                    pred = torch.cat((pred, pred0[i].unsqueeze(0)), 0)
                    label = torch.cat((label, GT[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss = 0
        else:
            self.loss = self.criterion_ce(pred, label)
        return self.loss



class Cls_Diag_predallendtoend(nn.Module):
    def __init__(self, opt):
        super(Cls_Diag_predallendtoend, self).__init__()
        self.opt = opt

        self.n_classes = 2
        if self.opt['TrainingSet'] == 'All':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.6])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_CDKN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.3, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_MGMT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_710 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_ATRX = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_EGFR = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.7, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_TERT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_OLIG2 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([45, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PDGFRA = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 10])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PTEN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 3.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_P53 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.opt['TrainingSet'] == 'TCGA':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 4.4])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_CDKN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_MGMT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_710 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_ATRX = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 3])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_EGFR = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_TERT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PDGFRA = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 10])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PTEN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 6.5])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_P53 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.6])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])

        elif self.opt['TrainingSet'] == 'Tiantan':
            self.criterion_ce_IDH = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1.4, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_1p19q = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_MGMT = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([4, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_ATRX = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_EGFR = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([9, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_PTEN = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_P53 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1.2])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
            self.criterion_ce_OLIG2 = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([45, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])

        self._fc2_IDH_1 = nn.Linear(512, self.n_classes)
        self._fc2_1p19q_1 = nn.Linear(512, self.n_classes)
        self._fc2_ATRX_1 = nn.Linear(512, self.n_classes)
        self._fc2_EGFR_1 = nn.Linear(512, self.n_classes)
        self._fc2_MGMT_1 = nn.Linear(512, self.n_classes)
        self._fc2_PTEN_1 = nn.Linear(512, self.n_classes)
        self._fc2_P53_1 = nn.Linear(512, self.n_classes)

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            self._fc2_CDKN_1 = nn.Linear(512, self.n_classes)
            self._fc2_710_1 = nn.Linear(512, self.n_classes)
            self._fc2_TERT_1 = nn.Linear(512, self.n_classes)
            self._fc2_PDGFRA_1 = nn.Linear(512, self.n_classes)
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            self._fc2_OLIG2_1 = nn.Linear(512, self.n_classes)

        # atten
        self.attention_IDH = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_IDH = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_IDH = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_IDH = nn.Linear(128, 1)
        self.attention_1p19q = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_1p19q = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_1p19q = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_1p19q = nn.Linear(128, 1)
        self.attention_MGMT = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_MGMT = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_MGMT = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_MGMT = nn.Linear(128, 1)
        self.attention_ATRX = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_ATRX = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_ATRX = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_ATRX = nn.Linear(128, 1)
        self.attention_EGFR = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_EGFR = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_EGFR = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_EGFR = nn.Linear(128, 1)
        self.attention_PTEN = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_PTEN = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_PTEN = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_PTEN = nn.Linear(128, 1)
        self.attention_P53 = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attention_V_P53 = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_U_P53 = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.attention_weights_P53 = nn.Linear(128, 1)
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            self.attention_CDKN = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.attention_V_CDKN = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
            self.attention_U_CDKN = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
            self.attention_weights_CDKN = nn.Linear(128, 1)
            self.attention_710 = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.attention_V_710 = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
            self.attention_U_710 = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
            self.attention_weights_710 = nn.Linear(128, 1)
            self.attention_TERT = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.attention_V_TERT = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
            self.attention_U_TERT = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
            self.attention_weights_TERT = nn.Linear(128, 1)
            self.attention_PDGFRA = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.attention_V_PDGFRA = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
            self.attention_U_PDGFRA = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
            self.attention_weights_PDGFRA = nn.Linear(128, 1)
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            self.attention_OLIG2 = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.attention_V_OLIG2 = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
            self.attention_U_OLIG2 = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
            self.attention_weights_OLIG2 = nn.Linear(128, 1)


    def forward(self, encoded):
        """
            encoded: [B, N,512]
        """
        ########################   IDH   ########################
        encoded_IDH=encoded
        A_V_IDH = self.attention_V_IDH(encoded_IDH)  # BxNx128
        A_U_IDH = self.attention_U_IDH(encoded_IDH)  # BxNx128
        A_encoded_IDH = self.attention_weights_IDH(A_V_IDH * A_U_IDH)  # BxNx1
        A_encoded_IDH = F.softmax(A_encoded_IDH, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_IDH.shape[0]):
            if i == 0:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_IDH_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.cat((Final_con_layer_IDH, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_IDH_new = torch.cat(
                    (encoded_IDH_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_IDH = encoded_IDH_new  # Bx512

        ########################   1p19q   ########################
        encoded_1p19q=encoded
        A_V_1p19q = self.attention_V_1p19q(encoded_1p19q)  # BxNx128
        A_U_1p19q = self.attention_U_1p19q(encoded_1p19q)  # BxNx128
        A_encoded_1p19q = self.attention_weights_1p19q(A_V_1p19q * A_U_1p19q)  # BxNx1
        A_encoded_1p19q = F.softmax(A_encoded_1p19q, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_1p19q.shape[0]):
            if i == 0:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_1p19q_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.cat((Final_con_layer_1p19q, torch.unsqueeze(Final_con_layer, 0)),
                                                  dim=0)  # BSxNx512
                encoded_1p19q_new = torch.cat(
                    (encoded_1p19q_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_1p19q = encoded_1p19q_new  # Bx512

        ########################   MGMT   ########################
        encoded_MGMT=encoded
        A_V_MGMT = self.attention_V_MGMT(encoded_MGMT)  # BxNx128
        A_U_MGMT = self.attention_U_MGMT(encoded_MGMT)  # BxNx128
        A_encoded_MGMT = self.attention_weights_MGMT(A_V_MGMT * A_U_MGMT)  # BxNx1
        A_encoded_MGMT = F.softmax(A_encoded_MGMT, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_MGMT.shape[0]):
            if i == 0:
                Final_con_layer = encoded_MGMT[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_MGMT[i], 1).expand(-1, encoded_MGMT[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_MGMT = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_MGMT_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_MGMT[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_MGMT[i], 1).expand(-1, encoded_MGMT[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_MGMT = torch.cat((Final_con_layer_MGMT, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_MGMT_new = torch.cat(
                    (encoded_MGMT_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_MGMT = encoded_MGMT_new  # Bx512

        ########################   ATRX   ########################
        encoded_ATRX=encoded
        A_V_ATRX = self.attention_V_ATRX(encoded_ATRX)  # BxNx128
        A_U_ATRX = self.attention_U_ATRX(encoded_ATRX)  # BxNx128
        A_encoded_ATRX = self.attention_weights_ATRX(A_V_ATRX * A_U_ATRX)  # BxNx1
        A_encoded_ATRX = F.softmax(A_encoded_ATRX, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_ATRX.shape[0]):
            if i == 0:
                Final_con_layer = encoded_ATRX[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_ATRX[i], 1).expand(-1, encoded_ATRX[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_ATRX = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_ATRX_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_ATRX[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_ATRX[i], 1).expand(-1, encoded_ATRX[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_ATRX = torch.cat((Final_con_layer_ATRX, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_ATRX_new = torch.cat(
                    (encoded_ATRX_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_ATRX = encoded_ATRX_new  # Bx512

        ########################   EGFR   ########################
        encoded_EGFR=encoded
        A_V_EGFR = self.attention_V_EGFR(encoded_EGFR)  # BxNx128
        A_U_EGFR = self.attention_U_EGFR(encoded_EGFR)  # BxNx128
        A_encoded_EGFR = self.attention_weights_EGFR(A_V_EGFR * A_U_EGFR)  # BxNx1
        A_encoded_EGFR = F.softmax(A_encoded_EGFR, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_EGFR.shape[0]):
            if i == 0:
                Final_con_layer = encoded_EGFR[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_EGFR[i], 1).expand(-1, encoded_EGFR[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_EGFR = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_EGFR_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_EGFR[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_EGFR[i], 1).expand(-1, encoded_EGFR[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_EGFR = torch.cat((Final_con_layer_EGFR, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_EGFR_new = torch.cat(
                    (encoded_EGFR_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_EGFR = encoded_EGFR_new  # Bx512
        ########################   PTEN   ########################
        encoded_PTEN=encoded
        A_V_PTEN = self.attention_V_PTEN(encoded_PTEN)  # BxNx128
        A_U_PTEN = self.attention_U_PTEN(encoded_PTEN)  # BxNx128
        A_encoded_PTEN = self.attention_weights_PTEN(A_V_PTEN * A_U_PTEN)  # BxNx1
        A_encoded_PTEN = F.softmax(A_encoded_PTEN, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_PTEN.shape[0]):
            if i == 0:
                Final_con_layer = encoded_PTEN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_PTEN[i], 1).expand(-1, encoded_PTEN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_PTEN = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_PTEN_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_PTEN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_PTEN[i], 1).expand(-1, encoded_PTEN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_PTEN = torch.cat((Final_con_layer_PTEN, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_PTEN_new = torch.cat(
                    (encoded_PTEN_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_PTEN = encoded_PTEN_new  # Bx512
        ########################   P53   ########################
        encoded_P53 = encoded
        A_V_P53 = self.attention_V_P53(encoded_P53)  # BxNx128
        A_U_P53 = self.attention_U_P53(encoded_P53)  # BxNx128
        A_encoded_P53 = self.attention_weights_P53(A_V_P53 * A_U_P53)  # BxNx1
        A_encoded_P53 = F.softmax(A_encoded_P53, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_P53.shape[0]):
            if i == 0:
                Final_con_layer = encoded_P53[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_P53[i], 1).expand(-1, encoded_P53[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_P53 = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_P53_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_P53[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_P53[i], 1).expand(-1, encoded_P53[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_P53 = torch.cat((Final_con_layer_P53, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_P53_new = torch.cat(
                    (encoded_P53_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_P53 = encoded_P53_new  # Bx512
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            ########################   CDKN   ########################
            encoded_CDKN = encoded
            A_V_CDKN = self.attention_V_CDKN(encoded_CDKN)  # BxNx128
            A_U_CDKN = self.attention_U_CDKN(encoded_CDKN)  # BxNx128
            A_encoded_CDKN = self.attention_weights_CDKN(A_V_CDKN * A_U_CDKN)  # BxNx1
            A_encoded_CDKN = F.softmax(A_encoded_CDKN, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_CDKN.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_CDKN[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_CDKN = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_CDKN_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_CDKN[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_CDKN = torch.cat((Final_con_layer_CDKN, torch.unsqueeze(Final_con_layer, 0)),
                                                     dim=0)  # BSxNx512
                    encoded_CDKN_new = torch.cat(
                        (encoded_CDKN_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_CDKN = encoded_CDKN_new  # Bx512
            ########################   710   ########################
            encoded_710 = encoded
            A_V_710 = self.attention_V_710(encoded_710)  # BxNx128
            A_U_710 = self.attention_U_710(encoded_710)  # BxNx128
            A_encoded_710 = self.attention_weights_710(A_V_710 * A_U_710)  # BxNx1
            A_encoded_710 = F.softmax(A_encoded_710, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_710.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_710[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_710[i], 1).expand(-1, encoded_710[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_710 = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_710_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_710[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_710[i], 1).expand(-1, encoded_710[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_710 = torch.cat((Final_con_layer_710, torch.unsqueeze(Final_con_layer, 0)),
                                                    dim=0)  # BSxNx512
                    encoded_710_new = torch.cat(
                        (encoded_710_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_710 = encoded_710_new  # Bx512
            ########################   TERT   ########################
            encoded_TERT = encoded
            A_V_TERT = self.attention_V_TERT(encoded_TERT)  # BxNx128
            A_U_TERT = self.attention_U_TERT(encoded_TERT)  # BxNx128
            A_encoded_TERT = self.attention_weights_TERT(A_V_TERT * A_U_TERT)  # BxNx1
            A_encoded_TERT = F.softmax(A_encoded_TERT, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_TERT.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_TERT[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_TERT[i], 1).expand(-1, encoded_TERT[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_TERT = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_TERT_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_TERT[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_TERT[i], 1).expand(-1, encoded_TERT[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_TERT = torch.cat((Final_con_layer_TERT, torch.unsqueeze(Final_con_layer, 0)),
                                                     dim=0)  # BSxNx512
                    encoded_TERT_new = torch.cat(
                        (encoded_TERT_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_TERT = encoded_TERT_new  # Bx512
            ########################   PDGFRA   ########################
            encoded_PDGFRA = encoded
            A_V_PDGFRA = self.attention_V_PDGFRA(encoded_PDGFRA)  # BxNx128
            A_U_PDGFRA = self.attention_U_PDGFRA(encoded_PDGFRA)  # BxNx128
            A_encoded_PDGFRA = self.attention_weights_PDGFRA(A_V_PDGFRA * A_U_PDGFRA)  # BxNx1
            A_encoded_PDGFRA = F.softmax(A_encoded_PDGFRA, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_PDGFRA.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_PDGFRA[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_PDGFRA[i], 1).expand(-1,
                                                                                  encoded_PDGFRA[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_PDGFRA = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_PDGFRA_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_PDGFRA[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_PDGFRA[i], 1).expand(-1,
                                                                                  encoded_PDGFRA[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_PDGFRA = torch.cat((Final_con_layer_PDGFRA, torch.unsqueeze(Final_con_layer, 0)),
                                                       dim=0)  # BSxNx512
                    encoded_PDGFRA_new = torch.cat(
                        (encoded_PDGFRA_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_PDGFRA = encoded_PDGFRA_new  # Bx512

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            ########################   OLIG2   ########################
            encoded_OLIG2 = encoded
            A_V_OLIG2 = self.attention_V_OLIG2(encoded_OLIG2)  # BxNx128
            A_U_OLIG2 = self.attention_U_OLIG2(encoded_OLIG2)  # BxNx128
            A_encoded_OLIG2 = self.attention_weights_OLIG2(A_V_OLIG2 * A_U_OLIG2)  # BxNx1
            A_encoded_OLIG2 = F.softmax(A_encoded_OLIG2, dim=1)[..., 0]  # BxN AMIL attention map
            for i in range(encoded_OLIG2.shape[0]):
                if i == 0:
                    Final_con_layer = encoded_OLIG2[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_OLIG2[i], 1).expand(-1, encoded_OLIG2[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_OLIG2 = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                    encoded_OLIG2_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
                else:
                    Final_con_layer = encoded_OLIG2[i]  # Nx512
                    saliency_map = torch.unsqueeze(A_encoded_OLIG2[i], 1).expand(-1, encoded_OLIG2[i].shape[1])  # Nx512
                    Final_con_layer = Final_con_layer * saliency_map  # Nx512
                    Final_con_layer_OLIG2 = torch.cat((Final_con_layer_OLIG2, torch.unsqueeze(Final_con_layer, 0)),
                                                      dim=0)  # BSxNx512
                    encoded_OLIG2_new = torch.cat(
                        (encoded_OLIG2_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
            encoded_OLIG2 = encoded_OLIG2_new  # Bx512
        logits_IDH = self._fc2_IDH_1(encoded_IDH)  # [BS,2]
        logits_1p19q = self._fc2_1p19q_1(encoded_1p19q)#[BS,2]
        logits_MGMT = self._fc2_MGMT_1(encoded_MGMT)  # [BS,2]
        logits_ATRX = self._fc2_ATRX_1(encoded_ATRX)  # [BS,2]
        logits_EGFR = self._fc2_EGFR_1(encoded_EGFR)  # [BS,2]
        logits_PTEN = self._fc2_PTEN_1(encoded_PTEN)  # [BS,2]
        logits_P53 = self._fc2_P53_1(encoded_P53)  # [BS,2]
        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'TCGA':
            logits_CDKN = self._fc2_CDKN_1(encoded_CDKN)  # [BS,2]
            logits_710 = self._fc2_710_1(encoded_710)  # [BS,2]
            logits_TERT = self._fc2_TERT_1(encoded_TERT)  # [BS,2]
            logits_PDGFRA = self._fc2_PDGFRA_1(encoded_PDGFRA)  # [BS,2]

        if self.opt['TrainingSet'] == 'All' or self.opt['TrainingSet'] == 'Tiantan':
            logits_OLIG2 = self._fc2_OLIG2_1(encoded_OLIG2)  # [BS,2]

        if self.opt['TrainingSet'] == 'All':
            results_dict = {'logits_IDH': logits_IDH,'logits_1p19q': logits_1p19q,'logits_CDKN': logits_CDKN,'logits_MGMT': logits_MGMT,
                            'logits_ATRX': logits_ATRX,'logits_EGFR': logits_EGFR,'logits_PTEN': logits_PTEN,'logits_P53': logits_P53,
                            'logits_710': logits_710,'logits_TERT': logits_TERT,'logits_PDGFRA': logits_PDGFRA,'logits_OLIG2': logits_OLIG2}

        if self.opt['TrainingSet'] == 'TCGA':
            results_dict = {'logits_IDH': logits_IDH,'logits_1p19q': logits_1p19q,'logits_CDKN': logits_CDKN,'logits_MGMT': logits_MGMT,
                            'logits_ATRX': logits_ATRX,'logits_EGFR': logits_EGFR,'logits_PTEN': logits_PTEN,'logits_P53': logits_P53,
                            'logits_710': logits_710,'logits_TERT': logits_TERT,'logits_PDGFRA': logits_PDGFRA}
        if self.opt['TrainingSet'] == 'Tiantan':
            results_dict = {'logits_IDH': logits_IDH,'logits_1p19q': logits_1p19q,'logits_MGMT': logits_MGMT,'logits_ATRX': logits_ATRX,
                            'logits_EGFR': logits_EGFR,'logits_PTEN': logits_PTEN,'logits_P53': logits_P53,'logits_OLIG2': logits_OLIG2}

        return results_dict

    def calculateLoss_IDH(self, pred, label):
        FLAT_normal = False
        self.loss_IDH = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_IDH = pred[i].unsqueeze(0)
                    label_IDH = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_IDH = torch.cat((pred_IDH, pred[i].unsqueeze(0)), 0)
                    label_IDH = torch.cat((label_IDH, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_IDH = 0
        else:
            self.loss_IDH = self.criterion_ce_IDH(pred_IDH, label_IDH)
        return self.loss_IDH

    def calculateLoss_1p19q(self, pred, label):
        FLAT_normal = False
        self.loss_1p19q = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_1p19q = pred[i].unsqueeze(0)
                    label_1p19q = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_1p19q = torch.cat((pred_1p19q, pred[i].unsqueeze(0)), 0)
                    label_1p19q = torch.cat((label_1p19q, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_1p19q = 0
        else:
            self.loss_1p19q = self.criterion_ce_1p19q(pred_1p19q, label_1p19q)
        return self.loss_1p19q

    def calculateLoss_CDKN(self, pred, label):
        FLAT_normal = False
        self.loss_CDKN = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_CDKN = pred[i].unsqueeze(0)
                    label_CDKN = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_CDKN = torch.cat((pred_CDKN, pred[i].unsqueeze(0)), 0)
                    label_CDKN = torch.cat((label_CDKN, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_CDKN = 0
        else:
            self.loss_CDKN = self.criterion_ce_CDKN(pred_CDKN, label_CDKN)
        return self.loss_CDKN

    def calculateLoss_MGMT(self, pred, label):
        FLAT_normal = False
        self.loss_MGMT = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_MGMT = pred[i].unsqueeze(0)
                    label_MGMT = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_MGMT = torch.cat((pred_MGMT, pred[i].unsqueeze(0)), 0)
                    label_MGMT = torch.cat((label_MGMT, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_MGMT = 0
        else:
            self.loss_MGMT = self.criterion_ce_MGMT(pred_MGMT, label_MGMT)
        return self.loss_MGMT

    def calculateLoss_ATRX(self, pred, label):
        FLAT_normal = False
        self.loss_ATRX = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_ATRX = pred[i].unsqueeze(0)
                    label_ATRX = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_ATRX = torch.cat((pred_ATRX, pred[i].unsqueeze(0)), 0)
                    label_ATRX = torch.cat((label_ATRX, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_ATRX = 0
        else:
            self.loss_ATRX = self.criterion_ce_ATRX(pred_ATRX, label_ATRX)
        return self.loss_ATRX

    def calculateLoss_EGFR(self, pred, label):
        FLAT_normal = False
        self.loss_EGFR = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_EGFR = pred[i].unsqueeze(0)
                    label_EGFR = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_EGFR = torch.cat((pred_EGFR, pred[i].unsqueeze(0)), 0)
                    label_EGFR = torch.cat((label_EGFR, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_EGFR = 0
        else:
            self.loss_EGFR = self.criterion_ce_EGFR(pred_EGFR, label_EGFR)
        return self.loss_EGFR

    def calculateLoss_PTEN(self, pred, label):
        FLAT_normal = False
        self.loss_PTEN = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_PTEN = pred[i].unsqueeze(0)
                    label_PTEN = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_PTEN = torch.cat((pred_PTEN, pred[i].unsqueeze(0)), 0)
                    label_PTEN = torch.cat((label_PTEN, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_PTEN = 0
        else:
            self.loss_PTEN = self.criterion_ce_PTEN(pred_PTEN, label_PTEN)
        return self.loss_PTEN

    def calculateLoss_TERT(self, pred, label):
        FLAT_normal = False
        self.loss_TERT = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_TERT = pred[i].unsqueeze(0)
                    label_TERT = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_TERT = torch.cat((pred_TERT, pred[i].unsqueeze(0)), 0)
                    label_TERT = torch.cat((label_TERT, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_TERT = 0
        else:
            self.loss_TERT = self.criterion_ce_TERT(pred_TERT, label_TERT)
        return self.loss_TERT

    def calculateLoss_P53(self, pred, label):
        FLAT_normal = False
        self.loss_P53 = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_P53 = pred[i].unsqueeze(0)
                    label_P53 = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_P53 = torch.cat((pred_P53, pred[i].unsqueeze(0)), 0)
                    label_P53 = torch.cat((label_P53, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_P53 = 0
        else:
            self.loss_P53 = self.criterion_ce_P53(pred_P53, label_P53)
        return self.loss_P53

    def calculateLoss_710(self, pred, label):
        FLAT_normal = False
        self.loss_710 = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_710 = pred[i].unsqueeze(0)
                    label_710 = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_710 = torch.cat((pred_710, pred[i].unsqueeze(0)), 0)
                    label_710 = torch.cat((label_710, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_710 = 0
        else:
            self.loss_710 = self.criterion_ce_710(pred_710, label_710)
        return self.loss_710

    def calculateLoss_PDGFRA(self, pred, label):
        FLAT_normal = False
        self.loss_PDGFRA = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_PDGFRA = pred[i].unsqueeze(0)
                    label_PDGFRA = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_PDGFRA = torch.cat((pred_PDGFRA, pred[i].unsqueeze(0)), 0)
                    label_PDGFRA = torch.cat((label_PDGFRA, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_PDGFRA = 0
        else:
            self.loss_PDGFRA = self.criterion_ce_PDGFRA(pred_PDGFRA, label_PDGFRA)
        return self.loss_PDGFRA

    def calculateLoss_OLIG2(self, pred, label):
        FLAT_normal = False
        self.loss_OLIG2 = 0
        count = 0
        for i in range(label.detach().cpu().numpy().shape[0]):
            if label.detach().cpu().numpy()[i] != 2:
                if count == 0:
                    pred_OLIG2 = pred[i].unsqueeze(0)
                    label_OLIG2 = label[i].unsqueeze(0)
                    count += 1
                else:
                    pred_OLIG2 = torch.cat((pred_OLIG2, pred[i].unsqueeze(0)), 0)
                    label_OLIG2 = torch.cat((label_OLIG2, label[i].unsqueeze(0)), 0)
                FLAT_normal = True
            else:
                continue

        if not FLAT_normal:
            self.loss_OLIG2 = 0
        else:
            self.loss_OLIG2 = self.criterion_ce_OLIG2(pred_OLIG2, label_OLIG2)
        return self.loss_OLIG2

class Cls_His_Grade_2016(nn.Module):
    def __init__(self, opt):
        super(Cls_His_Grade_2016, self).__init__()
        self.opt = opt

        self.n_classes_Grade = 3
        self.n_classes_His = 3
        self._fc2_His= nn.Linear(512, self.n_classes_His,bias=True)
        self._fc2_Grade = nn.Linear(512, self.n_classes_Grade,bias=True)
        self.attention_His = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_Grade = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_V_His = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_His = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_His = nn.Linear(128, 1)
        self.attention_V_Grade = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_Grade = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_Grade = nn.Linear(128, 1)



    def forward(self, encoded_His,encoded_Grade):
        """
            encoded_His: [B, N/N+1,512]
            encoded_Grade: [B, N/N+1,512]
        """
        encoded_His_ori=encoded_His
        encoded_Grade_ori = encoded_Grade
        A_V_His = self.attention_V_His(encoded_His)  # BxNx128
        A_U_His = self.attention_U_His(encoded_His)  # BxNx128
        A_encoded_His = self.attention_weights_His(A_V_His * A_U_His)  # BxNx1
        A_encoded_His = F.softmax(A_encoded_His, dim=1)[...,0]  # BxN AMIL attention map
        for i in range(encoded_His.shape[0]):
            if i==0:
                Final_con_layer=encoded_His[i] # Nx512
                saliency_map=torch.unsqueeze(A_encoded_His[i], 1).expand(-1, encoded_His[i].shape[1]) # Nx512
                Final_con_layer=Final_con_layer*saliency_map # Nx512
                Final_con_layer_His=torch.unsqueeze(Final_con_layer, 0)# 1xNx512
                encoded_His_new=torch.unsqueeze(torch.sum(Final_con_layer,dim=0), dim=0) # 1x512
            else:
                Final_con_layer = encoded_His[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_His[i], 1).expand(-1, encoded_His[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.cat((Final_con_layer_His,torch.unsqueeze(Final_con_layer, 0)),dim=0) # BSxNx512
                encoded_His_new=torch.cat((encoded_His_new,torch.unsqueeze(torch.sum(Final_con_layer,dim=0), dim=0) ), 0)
        encoded_His=encoded_His_new # Bx512

        A_V_Grade = self.attention_V_Grade(encoded_Grade)  # BxNx128
        A_U_Grade = self.attention_U_Grade(encoded_Grade)  # BxNx128
        A_encoded_Grade = self.attention_weights_Grade(A_V_Grade * A_U_Grade)  # BxNx1
        A_encoded_Grade = F.softmax(A_encoded_Grade, dim=1)[..., 0]  # BxN
        for i in range(encoded_Grade.shape[0]):
            if i == 0:
                Final_con_layer = encoded_Grade[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_Grade[i], 1).expand(-1, encoded_Grade[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_Grade = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_Grade_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_Grade[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_Grade[i], 1).expand(-1, encoded_Grade[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_Grade = torch.cat((Final_con_layer_Grade, torch.unsqueeze(Final_con_layer, 0)), dim=0)
                encoded_Grade_new = torch.cat((encoded_Grade_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_Grade = encoded_Grade_new  # Bx512
        ####################saliency maps for his
        weight_A=torch.unsqueeze(self._fc2_His.weight[0],dim=1) #[512,1]
        weight_O = torch.unsqueeze(self._fc2_His.weight[1],dim=1) #[512,1]
        weight_GBM = torch.unsqueeze(self._fc2_His.weight[2],dim=1) #[512,1]
        saliency_A=torch.matmul(Final_con_layer_His,weight_A)[...,0]  # [BSxN]
        saliency_O = torch.matmul(Final_con_layer_His, weight_O)[..., 0]  # [BSxN]
        saliency_GBM = torch.matmul(Final_con_layer_His, weight_GBM)[..., 0]  # [BSxN]
        if self._fc2_His.bias is not None:
            saliency_A=saliency_A+self._fc2_His.bias[0]/encoded_His_ori.shape[1] # [BSxN]
            saliency_O = saliency_O + self._fc2_His.bias[1] / encoded_His_ori.shape[1]# [BSxN]
            saliency_GBM = saliency_GBM + self._fc2_His.bias[2] / encoded_His_ori.shape[1]# [BSxN]
        ####################saliency maps for grade
        weight_G2 = torch.unsqueeze(self._fc2_Grade.weight[0], dim=1)  # [512,1]
        weight_G3 = torch.unsqueeze(self._fc2_Grade.weight[1], dim=1)  # [512,1]
        weight_G4 = torch.unsqueeze(self._fc2_Grade.weight[2], dim=1)  # [512,1]
        saliency_G2 = torch.matmul(Final_con_layer_Grade, weight_G2)[..., 0]  # [BSxN]
        saliency_G3 = torch.matmul(Final_con_layer_Grade, weight_G3)[..., 0]  # [BSxN]

        weight_G4_temp = np.array(weight_G4.tolist())[...,0]
        index = np.argsort(np.abs(weight_G4_temp))
        # weight_G4=weight_G4[list(index[int(512*0.5):]),:]
        # Final_con_layer_Grade=Final_con_layer_Grade[...,list(index[int(512*0.5):])]

        saliency_G4 = torch.matmul(Final_con_layer_Grade, weight_G4)[..., 0]  # [BSxN]


        if self._fc2_Grade.bias is not None:
            saliency_G2 = saliency_G2 + self._fc2_Grade.bias[0] / encoded_Grade_ori.shape[1]  # [BSxN]
            saliency_G3 = saliency_G3 + self._fc2_Grade.bias[1] / encoded_Grade_ori.shape[1]  # [BSxN]
            saliency_G4 = saliency_G4 + self._fc2_Grade.bias[2] / encoded_Grade_ori.shape[1]  # [BSxN]

        logits_His = self._fc2_His(encoded_His)  # [BS,3]
        logits_Grade = self._fc2_Grade(encoded_Grade)  # [BS,3]
        results_dict = {'logits_His': logits_His,'logits_Grade': logits_Grade}

        return  results_dict,saliency_A,saliency_O,saliency_GBM,saliency_G2,saliency_G3,saliency_G4

    def Loss_mutual_correlation(self,weight_IDH_wt,weight_His_GBM,weight_1p19q_codel,weight_His_O,epoch,IDH_only=False):
        """
        shape:torch[2500]
        """
        weight_IDH_wt=weight_IDH_wt.tolist()
        weight_His_GBM=weight_His_GBM.tolist()
        x=weight_IDH_wt
        b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
        Index_IDH_wt = [x[0] for x in b]
        x = weight_His_GBM
        b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
        Index_His_GBM = [x[0] for x in b]

        if not IDH_only:
            weight_1p19q_codel = weight_1p19q_codel.tolist()
            weight_His_O = weight_His_O.tolist()
            x = weight_1p19q_codel
            b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
            Index_1p19q_codel = [x[0] for x in b]
            x = weight_His_O
            b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
            Index_His_O = [x[0] for x in b]

        self.opt['Network']['top_K_patch'] = int(self.opt['fixdim'] / 3)
        top_K_patch = int(self.opt['Network']['top_K_patch'] * (0.85 ** (int(epoch / 10))))
        #### IDH-wt  **** GBM
        loss_IDH_GBM=0
        for i in range(top_K_patch):
            index_patch_low=Index_IDH_wt[i]
            if i<=int(self.opt['Network']['top_K_patch']/2):
                target_low_index_list=Index_His_GBM[0:self.opt['Network']['top_K_patch']]
            else:
                target_low_index_list=Index_His_GBM[i-int(self.opt['Network']['top_K_patch']/2):i+int(self.opt['Network']['top_K_patch']/2)]
            if not index_patch_low in target_low_index_list:
                loss_IDH_GBM+=1
        loss_IDH_GBM=loss_IDH_GBM/top_K_patch
        loss_GBM_IDH = 0
        for i in range(top_K_patch):
            index_patch_low = Index_His_GBM[i]
            if i <= int(self.opt['Network']['top_K_patch'] / 2):
                target_low_index_list = Index_IDH_wt[0:self.opt['Network']['top_K_patch']]
            else:
                target_low_index_list = Index_IDH_wt[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(self.opt['Network']['top_K_patch'] / 2)]
            if not index_patch_low in target_low_index_list:
                loss_GBM_IDH += 1
        loss_GBM_IDH = loss_GBM_IDH / top_K_patch
        loss_IDH_GBM=(loss_GBM_IDH+loss_IDH_GBM)/2
        #### 1p19q codel  **** O
        if not IDH_only:
            loss_1p19q_O=0
            for i in range(top_K_patch):
                index_patch_low=Index_1p19q_codel[i]
                if i<=int(self.opt['Network']['top_K_patch']/2):
                    target_low_index_list=Index_His_O[0:self.opt['Network']['top_K_patch']]
                else:
                    target_low_index_list=Index_His_O[i-int(self.opt['Network']['top_K_patch']/2):i+int(self.opt['Network']['top_K_patch']/2)]
                if not index_patch_low in target_low_index_list:
                    loss_1p19q_O+=1
            loss_1p19q_O=loss_1p19q_O/top_K_patch
            loss_O_1p19q = 0
            for i in range(top_K_patch):
                index_patch_low = Index_His_O[i]
                if i <= int(self.opt['Network']['top_K_patch'] / 2):
                    target_low_index_list = Index_1p19q_codel[0:self.opt['Network']['top_K_patch']]
                else:
                    target_low_index_list = Index_1p19q_codel[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(self.opt['Network']['top_K_patch'] / 2)]
                if not index_patch_low in target_low_index_list:
                    loss_O_1p19q += 1
            loss_O_1p19q = loss_O_1p19q / top_K_patch
            loss_1p19q_O=(loss_O_1p19q+loss_1p19q_O)/2


        if IDH_only:
            loss_mutual_correlation=loss_IDH_GBM
        else:
            loss_mutual_correlation = loss_IDH_GBM + loss_1p19q_O
        loss_mutual_correlation=torch.from_numpy(np.array([loss_mutual_correlation])).cuda(self.opt['gpus'][0])[0]
        return loss_mutual_correlation



def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
if __name__ == "__main__":
    import argparse
    # import h5py
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--opt', type=str, default='./config/mine.yml')
    # args = parser.parse_args()
    # with open(args.opt) as f:
    #     opt = yaml.load(f, Loader=SafeLoader)
    #
    # res_init = Mine_init(opt).cuda(opt['gpus'][0])
    # res_IDH=Mine_IDH(opt).cuda(opt['gpus'][0])
    # res_1p19q = Mine_1p19q(opt).cuda(opt['gpus'][0])
    # res_CDKN = Mine_CDKN(opt).cuda(opt['gpus'][0])
    # res_Graph = Label_correlation_Graph(opt).cuda(opt['gpus'][0])
    # res_His = Mine_His(opt).cuda(opt['gpus'][0])
    # res_Grade = Mine_Grade(opt).cuda(opt['gpus'][0])
    # res_Cls_His_Grade = Cls_His_Grade(opt).cuda(opt['gpus'][0])
    #
    # init_weights(res_init, init_type='xavier', init_gain=1)
    # init_weights(res_IDH, init_type='xavier', init_gain=1)
    # init_weights(res_1p19q, init_type='xavier', init_gain=1)
    # init_weights(res_CDKN, init_type='xavier', init_gain=1)
    # init_weights(res_His, init_type='xavier', init_gain=1)
    # init_weights(res_Grade, init_type='xavier', init_gain=1)
    # device = torch.device('cuda:{}'.format(opt['gpus'][0]))
    # res_init.to(device)
    # res_IDH.to(device)
    # res_1p19q.to(device)
    # res_CDKN.to(device)
    # res_Graph.to(device)
    # res_His.to(device)
    # res_Grade.to(device)
    # res_Cls_His_Grade.to(device)
    #
    # # input1 = torch.ones((8, 2500,1024)).cuda(opt['gpus'][0])
    # root = opt['dataDir'] + 'Res50_feature_2500_fixdim0/'
    # # root=r'D:\PhD\Project_WSI\data\Res50_feature_2500/'
    # patch_all0 =torch.from_numpy(np.array(h5py.File(root + 'TCGA-DU-A5TY-01Z-00-DX1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])# (1,N,1024)
    # patch_all1=torch.from_numpy(np.array(h5py.File(root + 'TCGA-HT-8104-01A-01-TS1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])# (1,N,1024)
    # patch_all2 = torch.from_numpy(np.array(h5py.File(root + 'TCGA-CS-6188-01A-01-BS1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])  # (1,N,1024)
    # patch_all3 = torch.from_numpy(np.array(h5py.File(root + 'TCGA-DU-7010-01Z-00-DX1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])  # (1,N,1024)
    # input1 = torch.cat((patch_all0, patch_all1, patch_all2,patch_all3), 0)  # [4,N,1024]
    #
    # hidden_states_init = res_init(input1)
    #
    # hidden_states, encoded_IDH=res_IDH(hidden_states_init)
    # hidden_states, encoded_1p19q = res_1p19q(hidden_states)
    # encoded_CDKN = res_CDKN(hidden_states)
    # # a_max = np.max(hidden_states.detach().numpy()[0])
    # # a_min = np.min(hidden_states.detach().numpy()[0])
    #
    # out,weight_IDH_wt,weight_1p19q_codel,encoded_IDH0, encoded_1p19q0, encoded_CDKN0 = res_Graph(encoded_IDH,encoded_1p19q,encoded_CDKN)
    # loss_IDH = res_Graph.calculateLoss_IDH(out['logits_IDH'], torch.from_numpy(np.array([1,1,1,1])).cuda(opt['gpus'][0]))
    # loss_Graph=res_Graph.calculateLoss_Graph(encoded_IDH0, encoded_1p19q0, encoded_CDKN0)
    # hidden_states, encoded_His = res_His(hidden_states_init)
    # encoded_Grade=res_Grade(hidden_states)
    # out,weight_His_GBM,weight_His_O=res_Cls_His_Grade(encoded_His,encoded_Grade)
    #
    # loss_mutual_correlation=res_Cls_His_Grade.Loss_mutual_correlation(weight_IDH_wt, weight_1p19q_codel, weight_His_GBM, weight_His_O, 0)
    # a=1
































