import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import os
import torch.cuda
import argparse, time, random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy import interp
""" curriculum learning based training strategy """
import cv2

class CL_strategy():
    def __init__(self):
        super(CL_strategy, self).__init__()

def subgroup_label_gene(subtype,grade,IDH,pq, CDKN ):

    ###  WHO 2007 subtype of A/O/GBM
    if subtype == 'oligoastrocytoma':
        label_subtype = 3
    elif subtype == 'astrocytoma':
        label_subtype = 0
    elif subtype == 'oligodendroglioma':
        label_subtype = 1
    elif subtype == 'glioblastoma':
        label_subtype = 2
    else:
        label_subtype=4

    ###  WHO 2007 Grading
    if grade == 'G2':
        label_Grade = 0
    elif grade == 'G3':
        label_Grade = 1
    elif grade == 'G4':
        label_Grade = 2
    else:
        label_Grade = 3

    ###  molecular
    if IDH == 'WT':
        label_IDH = 0
    elif IDH == 'Mutant':
        label_IDH = 1
    else:
        label_IDH = 2
    if pq == 'non-codel':
        label_1p19q = 0
    elif pq == 'codel':
        label_1p19q = 1
    else:
        label_1p19q = 2
    if CDKN == -2 or CDKN == -1:
        label_CDKN = 1
    elif CDKN == 1 or CDKN == 0:
        label_CDKN = 0
    else:
        label_CDKN = 2

    ###  Diag: GBM,  G4 A, G3 A, G2 A, G3 O, G2 O, NA--> 0,1,2,3,4,5,6
    if label_IDH == 0:
        label_Diag = 0  # G4 GBM
    elif label_IDH == 1:
        if label_1p19q == 1:
            if label_Grade == 0:
                label_Diag = 5  # G2 Oligo
            else:
                label_Diag = 4  # G3 Oligo
        elif label_1p19q == 0:
            if label_CDKN == 1 or label_Grade == 2:
                label_Diag = 1  # G4 A
            elif label_CDKN == 0:
                if label_Grade == 0:
                    label_Diag = 3  # G2 A
                else:
                    label_Diag = 2  # G3 A
            elif label_CDKN == 2:
                label_Diag = 6
        elif label_1p19q == 2:
            label_Diag = 6
    elif label_IDH == 2:
        label_Diag = 6

    if label_Diag!=6:
        if label_subtype==2:
            subtype2007='G'
        if label_subtype==0:
            subtype2007='A'
        if label_subtype==1:
            subtype2007='O'
        if label_subtype==3:
            subtype2007='OA'
        if label_Grade==2:
            grade2007='G4'
        if label_Grade==1:
            grade2007='G3'
        if label_Grade==0:
            grade2007='G2'
        if (label_Grade==0 or label_Grade==1) and  (label_Diag==0 or label_Diag==1):
            gradechage=1
        else:
            gradechage=0
        return  subtype2007 ,grade2007,gradechage, True
    else:
        return None, None, None, False

""" spatial_pyramid_pooling """
import math
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size=[]):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp

def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)







class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
def saliency_comparison(saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4, pred_his,pred_garde):
    for i in range(saliency_A.shape[0]):
        pred_his=pred_his[i]
        saliency_A_max = np.max(saliency_A[i].detach().cpu().numpy())
        saliency_A_min = np.min(saliency_A[i].detach().cpu().numpy())
        saliency_A_mean = np.mean(saliency_A[i].detach().cpu().numpy())
        saliency_A_vis=saliency_A[i].detach().cpu().numpy()

        saliency_O_max = np.max(saliency_O[i].detach().cpu().numpy())
        saliency_O_min = np.min(saliency_O[i].detach().cpu().numpy())
        saliency_O_mean = np.mean(saliency_O[i].detach().cpu().numpy())
        saliency_O_vis=saliency_O[i].detach().cpu().numpy()

        saliency_GBM_max = np.max(saliency_GBM[i].detach().cpu().numpy())
        saliency_GBM_min = np.min(saliency_GBM[i].detach().cpu().numpy())
        saliency_GBM_mean = np.mean(saliency_GBM[i].detach().cpu().numpy())
        saliency_GBM_vis=saliency_GBM[i].detach().cpu().numpy()

        mean=[saliency_A_mean,saliency_O_mean,saliency_GBM_mean]
        saliency_A_vis[saliency_A_vis < mean[pred_his]] = saliency_A_min
        saliency_A_vis = (saliency_A_vis - saliency_A_min) / (saliency_A_max - saliency_A_min)
        saliency_O_vis[saliency_O_vis < mean[pred_his]] = saliency_O_min
        saliency_O_vis = (saliency_O_vis - saliency_O_min) / (saliency_O_max - saliency_O_min)
        saliency_GBM_vis[saliency_GBM_vis < mean[pred_his]] = saliency_GBM_min
        saliency_GBM_vis = (saliency_GBM_vis - saliency_GBM_min) / (saliency_GBM_max - saliency_GBM_min)

        saliency_final_His=np.zeros(shape=saliency_O[i].shape[0])

        if pred_his==0:
            saliency_final_His[saliency_A_vis>0]=1
            saliency_final_His[saliency_O_vis > 0] = 0.5
            saliency_final_His[saliency_GBM_vis > 0] = 0.5
        elif pred_his==1:
            saliency_final_His[saliency_O_vis > 0] = 1
            saliency_final_His[saliency_A_vis > 0] = 0.5
            saliency_final_His[saliency_GBM_vis > 0] = 0.5
        else:
            saliency_final_His[saliency_GBM_vis > 0] = 1
            saliency_final_His[saliency_O_vis > 0] = 0.5
            saliency_final_His[saliency_A_vis > 0] = 0.5

        pred_garde = pred_garde[i]
        saliency_G2_max = np.max(saliency_G2[i].detach().cpu().numpy())
        saliency_G2_min = np.min(saliency_G2[i].detach().cpu().numpy())
        saliency_G2_mean = np.mean(saliency_G2[i].detach().cpu().numpy())
        saliency_G2_vis = saliency_G2[i].detach().cpu().numpy()

        saliency_G3_max = np.max(saliency_G3[i].detach().cpu().numpy())
        saliency_G3_min = np.min(saliency_G3[i].detach().cpu().numpy())
        saliency_G3_mean = np.mean(saliency_G3[i].detach().cpu().numpy())
        saliency_G3_vis = saliency_G3[i].detach().cpu().numpy()

        saliency_G4_max = np.max(saliency_G4[i].detach().cpu().numpy())
        saliency_G4_min = np.min(saliency_G4[i].detach().cpu().numpy())
        saliency_G4_mean = np.mean(saliency_G4[i].detach().cpu().numpy())
        saliency_G4_vis = saliency_G4[i].detach().cpu().numpy()

        mean = [saliency_G2_mean, saliency_G3_mean, saliency_G4_mean]
        saliency_G2_vis[saliency_G2_vis < mean[pred_garde]] = saliency_G2_min
        saliency_G2_vis = (saliency_G2_vis - saliency_G2_min) / (saliency_G2_max - saliency_G2_min)
        saliency_G3_vis[saliency_G3_vis < mean[pred_garde]] = saliency_G3_min
        saliency_G3_vis = (saliency_G3_vis - saliency_G3_min) / (saliency_G3_max - saliency_G3_min)
        saliency_G4_vis[saliency_G4_vis < mean[pred_garde]] = saliency_G4_min
        saliency_G4_vis = (saliency_G4_vis - saliency_G4_min) / (saliency_G4_max - saliency_G4_min)

        saliency_final_Grade = np.zeros(shape=saliency_G3[i].shape[0])

        if pred_garde == 0:
            saliency_final_Grade[saliency_G2_vis > 0] = 1
            saliency_final_Grade[saliency_G3_vis > 0] = 0.5
            saliency_final_Grade[saliency_G4_vis > 0] = 0.5
        elif pred_garde == 1:
            saliency_final_Grade[saliency_G3_vis > 0] = 1
            saliency_final_Grade[saliency_G2_vis > 0] = 0.5
            saliency_final_Grade[saliency_G4_vis > 0] = 0.5
        else:
            saliency_final_Grade[saliency_G4_vis > 0] = 1
            saliency_final_Grade[saliency_G3_vis > 0] = 0.5
            saliency_final_Grade[saliency_G2_vis > 0] = 0.5

        return saliency_final_His,saliency_final_Grade

def saliency_map_read(opt,file_name,epoch,opt_name=None):
    if opt_name!=None:
        opt_name0=opt_name
    else:
        opt_name0=opt['name']
    if epoch<=opt['saliency_ep'][0]: # init
        for i in range(len(file_name)):
            if i==0:
                saliency_map_His = np.expand_dims(np.load('./saliency/init/'+opt_name0+'/His/' + file_name[i] + '.npy', allow_pickle=True),0)
                saliency_map_Grade = np.expand_dims(np.load('./saliency/init/'+opt_name0+'/Grade/' + file_name[i] + '.npy', allow_pickle=True), 0)
            else:
                saliency_map_His = np.concatenate((saliency_map_His,np.expand_dims(np.load('./saliency/init/'+opt_name0+'/His/' + file_name[i] + '.npy', allow_pickle=True),0)),0)
                saliency_map_Grade = np.concatenate((saliency_map_Grade, np.expand_dims(
                    np.load('./saliency/init/'+opt_name0+'/Grade/' + file_name[i] + '.npy', allow_pickle=True), 0)), 0)

    else:
        for i in range(len(file_name)):
            if i == 0:
                saliency_map_His = np.expand_dims(
                    np.load('./saliency/dynamic/'+opt_name0+'/His/' + file_name[i] + '.npy', allow_pickle=True), 0)
                saliency_map_Grade = np.expand_dims(
                    np.load('./saliency/dynamic/'+opt_name0+'/Grade/' + file_name[i] + '.npy', allow_pickle=True), 0)
            else:
                saliency_map_His = np.concatenate((saliency_map_His, np.expand_dims(
                    np.load('./saliency/dynamic/'+opt_name0+'/His/' + file_name[i] + '.npy', allow_pickle=True), 0)), 0)
                saliency_map_Grade = np.concatenate((saliency_map_Grade, np.expand_dims(
                    np.load('./saliency/dynamic/'+opt_name0+'/Grade/' + file_name[i] + '.npy', allow_pickle=True), 0)), 0)

    return saliency_map_His,saliency_map_Grade


def generate_saliency(opt,Mine_model_init,Mine_model_His,Mine_model_Cls,trainDataset,testLoader,gpuID,epoch):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()

    if not os.path.exists('./saliency/dynamic/'+opt['name']+'/Grade/'):
        os.makedirs('./saliency/dynamic/'+opt['name']+'/Grade/')
    # else:
    #     remove_all_file('./saliency/dynamic/'+opt['name']+'/Grade/')
    if not os.path.exists('./saliency/dynamic/'+opt['name']+'/His/'):
        os.makedirs('./saliency/dynamic/'+opt['name']+'/His/')
    # else:
    #     remove_all_file('./saliency/dynamic/'+opt['name']+'/His/')
    trainLoader = DataLoader(trainDataset, batch_size=1,num_workers=8, shuffle=True)


    train_bar = tqdm(trainLoader)
    count = 0
    for packs in train_bar:
        img = packs[0]
        file_name = packs[2]
        count += 1
        img = img.cuda(gpuID[0])

        saliency_map_His, saliency_map_Grade = saliency_map_read(opt, file_name, epoch)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])


        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature, saliency_map_His, saliency_map_Grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
            encoded_His, encoded_Grade)
        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2
        ################################ VISUALIZATION ################################
        saliency_final_His,saliency_final_Grade=saliency_comparison(saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4, pred_His,
                        pred_Grade)
        np.save('./saliency/dynamic/'+opt['name']+'/Grade/' + file_name[0] + '.npy', saliency_final_Grade)
        np.save('./saliency/dynamic/'+opt['name']+'/His/' + file_name[0] +'.npy', saliency_final_His)
    test_bar = tqdm(testLoader)
    count = 0
    for packs in test_bar:
        img = packs[0]
        file_name = packs[2]
        count += 1
        img = img.cuda(gpuID[0])
        saliency_map_His, saliency_map_Grade = saliency_map_read(opt, file_name, epoch)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])

        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature, saliency_map_His,
                                                                                            saliency_map_Grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
            encoded_His, encoded_Grade)
        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2
        ################################ VISUALIZATION ################################
        saliency_final_His, saliency_final_Grade = saliency_comparison(saliency_A, saliency_O, saliency_GBM,
                                                                       saliency_G2, saliency_G3, saliency_G4, pred_His,
                                                                       pred_Grade)
        np.save('./saliency/dynamic/'+opt['name']+'/Grade/' + file_name[0] + '.npy', saliency_final_Grade)
        np.save('./saliency/dynamic/'+opt['name']+'/His/' + file_name[0] + '.npy', saliency_final_His)




def validation_2016(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, dataloader, gpuID,epoch,opt_name=None):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()


    if 1:

        count_His = 0
        count_Grade = 0
        correct_His = 0
        correct_Grade=0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_His = []
        predicted_all_His = []

        G2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        G3_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        G4_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        all_metrics_Grade = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_Grade = []
        predicted_all_Grade = []

    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    count_noiseclean=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name=packs[2]

        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]


        # imp_his, imp_grade = imp_gene(opt,img)
        saliency_map_His, saliency_map_Grade = saliency_map_read(opt, file_name, epoch,opt_name=opt_name)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,saliency_map_His, saliency_map_Grade)
        results_dict,saliency_A,saliency_O,saliency_GBM,saliency_G2,saliency_G3,saliency_G4= Mine_model_Cls(encoded_His, encoded_Grade)

        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
        gt_His = label_his.tolist() #[BS] A  O GBM//0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2
        gt_Grade = label_grade.tolist()  # [BS] A  O GBM//0 1 2


        ################################ VISUALIZATION ################################
        #
        # saliency_final_His,saliency_final_Grade=saliency_comparison(saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4, pred_His,
        #                 pred_Grade)
        # np.save('./saliency/init/Grade/' + file_name[0] + '.npy', saliency_final_Grade)
        # np.save('./saliency/init/His/' + file_name[0] +'.npy', saliency_final_His)

        ################################ VISUALIZATION ################################


        for j in range(bs):
            ##################   His
            # A
            if gt_His[j] == 0:
                if pred_His[j] == 0:
                    A_metrics['tp'] += 1
                else:
                    A_metrics['fn'] += 1
            else:
                if not pred_His[j] == 0:
                    A_metrics['tn'] += 1
                else:
                    A_metrics['fp'] += 1
            # O
            if gt_His[j] == 1:
                if pred_His[j] == 1:
                    O_metrics['tp'] += 1
                else:
                    O_metrics['fn'] += 1
            else:
                if not pred_His[j] == 1:
                    O_metrics['tn'] += 1
                else:
                    O_metrics['fp'] += 1
            # GBM
            if gt_His[j] == 2:
                if pred_His[j] == 2:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred_His[j] == 2:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
            label_all_His.append(gt_His[j])
            predicted_all_His.append(pred_His_ori.detach().cpu().numpy()[j])
            count_His += 1
            if gt_His[j] == pred_His[j]:
                correct_His += 1
            ##################   Grade
            # G2
            if gt_Grade[j] == 0:
                if pred_Grade[j] == 0:
                    G2_metrics['tp'] += 1
                else:
                    G2_metrics['fn'] += 1
            else:
                if not pred_Grade[j] == 0:
                    G2_metrics['tn'] += 1
                else:
                    G2_metrics['fp'] += 1
            # G3
            if gt_Grade[j] == 1:
                if pred_Grade[j] == 1:
                    G3_metrics['tp'] += 1
                else:
                    G3_metrics['fn'] += 1
            else:
                if not pred_Grade[j] == 1:
                    G3_metrics['tn'] += 1
                else:
                    G3_metrics['fp'] += 1
            # G4
            if gt_Grade[j] == 2:
                if pred_Grade[j] == 2:
                    G4_metrics['tp'] += 1
                else:
                    G4_metrics['fn'] += 1
            else:
                if not pred_Grade[j] == 2:
                    G4_metrics['tn'] += 1
                else:
                    G4_metrics['fp'] += 1
            label_all_Grade.append(gt_Grade[j])
            predicted_all_Grade.append(pred_Grade_ori.detach().cpu().numpy()[j])
            count_Grade += 1
            if gt_Grade[j] == pred_Grade[j]:
                correct_Grade += 1


    ################################################ His
    Acc_His = correct_His / count_His

    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics_His['sen'] = (A_metrics['sen'] +  O_metrics['sen'] +
                          GBM_metrics['sen'] ) / 3
    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp']+0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp']+0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp']+0.000001)
    all_metrics_His['spec'] = (A_metrics['spec'] + O_metrics['spec'] +
                           GBM_metrics['spec'] ) / 3
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp']+0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp']+0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp']+0.000001)
    all_metrics_His['pre'] = (A_metrics['pre']  + O_metrics['pre'] +
                          GBM_metrics['pre'] ) / 3
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics_His['recall'] = (A_metrics['recall']  + O_metrics['recall'] +
                             GBM_metrics['recall'] ) / 3
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
                A_metrics['pre'] + A_metrics['recall']+0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
                O_metrics['pre'] + O_metrics['recall']+0.000001)
    GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (GBM_metrics['pre'] + GBM_metrics['recall']+0.000001)
    all_metrics_His['f1'] = (A_metrics['f1']  + O_metrics['f1'] +
                          GBM_metrics['f1']) / 3
    # AUC
    out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
    label_all_np = np.array(label_all_His)
    label_all_onehot = make_one_hot(label_all_np)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    all_metrics_His['AUC'] = roc_auc["macro"]

    ################################################ Grade
    Acc_Grade = correct_Grade / count_Grade

    #  Sensitivity
    G2_metrics['sen'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['sen'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['sen'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['sen'] = (G2_metrics['sen'] + G3_metrics['sen'] +
                                G4_metrics['sen']) / 3
    #  Spec
    G2_metrics['spec'] = (G2_metrics['tn']) / (G2_metrics['tn'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['spec'] = (G3_metrics['tn']) / (G3_metrics['tn'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['spec'] = (G4_metrics['tn']) / (G4_metrics['tn'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['spec'] = (G2_metrics['spec'] + G3_metrics['spec'] +
                                 G4_metrics['spec']) / 3
    #  Precision
    G2_metrics['pre'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['pre'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['pre'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['pre'] = (G2_metrics['pre'] + G3_metrics['pre'] +
                                G4_metrics['pre']) / 3
    #  Recall
    G2_metrics['recall'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['recall'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['recall'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['recall'] = (G2_metrics['recall'] + G3_metrics['recall'] +
                                   G4_metrics['recall']) / 3
    #  F1
    G2_metrics['f1'] = (2 * G2_metrics['pre'] * G2_metrics['recall']) / (
            G2_metrics['pre'] + G2_metrics['recall'] + 0.000001)
    G3_metrics['f1'] = (2 * G3_metrics['pre'] * G3_metrics['recall']) / (
            G3_metrics['pre'] + G3_metrics['recall'] + 0.000001)
    G4_metrics['f1'] = (2 * G4_metrics['pre'] * G4_metrics['recall']) / (
                G4_metrics['pre'] + G4_metrics['recall'] + 0.000001)
    all_metrics_Grade['f1'] = (G2_metrics['f1'] + G3_metrics['f1'] +
                               G4_metrics['f1']) / 3
    # AUC
    out_cls_all_softmax_Grade = F.softmax(torch.from_numpy(np.array(predicted_all_Grade)), dim=1).numpy()
    label_all_np = np.array(label_all_Grade)
    label_all_onehot = make_one_hot(label_all_np)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Grade[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    all_metrics_Grade['AUC'] = roc_auc["macro"]


    list_His = ( Acc_His, all_metrics_His['sen'], all_metrics_His['spec'],all_metrics_His['pre'],all_metrics_His['recall']
                 , all_metrics_His['f1'] ,all_metrics_His['AUC'])
    list_Grade = (
    Acc_Grade, all_metrics_Grade['sen'], all_metrics_Grade['spec'], all_metrics_Grade['pre'], all_metrics_Grade['recall']
    , all_metrics_Grade['f1'], all_metrics_Grade['AUC'])

    return list_His,list_Grade

def validation_2021(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,Mine_model_molecular,Mine_model_Graph, dataLoader, epoch):
     Mine_model_init.eval()
     Mine_model_His.eval()
     Mine_model_Cls.eval()
     Mine_model_molecular.eval()
     Mine_model_Graph.eval()
     gpuID=opt['gpus']
     if 1:
         count_IDH = 0
         count_1p19q = 0
         count_CDKN = 0
         count_Diag = 0
         count_Diag_Sim = 0
         correct_IDH = 0
         correct_1p19q = 0
         correct_CDKN = 0
         correct_Diag = 0
         correct_Diag_Sim = 0
         IDH_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         p19q_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         CDKN_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_GBM= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_G4A= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_G3A= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_G2A= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_G3O= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_G2O= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         Diag_all= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         DiagSim_GBM= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         DiagSim_G4A= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         DiagSim_G23A= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         DiagSim_G23O= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
         DiagSim_all= {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}

         label_all_IDH = []
         predicted_all_IDH = []
         label_all_1p19q = []
         predicted_all_1p19q = []
         label_all_CDKN = []
         predicted_all_CDKN = []
         label_all_Diag = []
         predicted_all_Diag = []
         label_all_DiagSim = []
         predicted_all_DiagSim = []

     test_bar = tqdm(dataLoader)


     for packs in test_bar:
         img = packs[0]
         label = packs[1]
         file_name = packs[2]

         img = img.cuda(gpuID[0])
         label = label.cuda(gpuID[0])
         label_IDH = label[:, 2]
         label_1p19q = label[:, 3]
         label_CDKN = label[:, 4]
         label_Diag_simple = label[:, 5]
         label_Diag = label[:, 6]

         saliency_map_His, saliency_map_Grade = saliency_map_read_stage2(opt, file_name)
         saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
         saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
         init_feature = Mine_model_init(img)  # (BS,2500,1024)
         hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,saliency_map_His,saliency_map_Grade)
         results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(encoded_His, encoded_Grade)

         ### WHO2007 prediction
         pred_His_ori = results_dict['logits_His']
         pred_Grade_ori = results_dict['logits_Grade']
         _, pred_His0 = torch.max(pred_His_ori.data, 1)
         pred_His = pred_His0.tolist()  # [BS] A  O GBM //0 1 2
         _, pred_Grade0 = torch.max(pred_Grade_ori.data, 1)
         pred_Grade = pred_Grade0.tolist()  # [BS] A  O GBM //0 1 2
         pred_His_ori = F.softmax(pred_His_ori)
         pred_Grade_ori = F.softmax(pred_Grade_ori)

         ### molecular prediction
         encoded_IDH, encoded_1p19q, encoded_CDKN = Mine_model_molecular(init_feature)
         results_dict, saliency_IDH_wt, saliency_1p19q_codel, encoded_IDH0, encoded_1p19q0, encoded_CDKN0 = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)
         pred_IDH_ori = results_dict['logits_IDH']
         pred_1p19q_ori = results_dict['logits_1p19q']
         pred_CDKN_ori = results_dict['logits_CDKN']
         _, pred_IDH0 = torch.max(pred_IDH_ori.data, 1)
         pred_IDH = pred_IDH0.tolist()
         gt_IDH  = label_IDH.tolist()
         _, pred_1p19q0 = torch.max(pred_1p19q_ori.data, 1)
         pred_1p19q = pred_1p19q0.tolist()
         gt_1p19q  = label_1p19q .tolist()
         _, pred_CDKN0 = torch.max(pred_CDKN_ori.data, 1)
         pred_CDKN = pred_CDKN0.tolist()
         gt_CDKN = label_CDKN.tolist()

         ### Diag prediction
         gt_Diag=label_Diag.tolist()
         Diag0=Diag_full(pred_IDH0,pred_1p19q0,pred_CDKN0,pred_His0,pred_Grade0)
         pred_Diag=Diag0.tolist()

         ### Diag Simple prediction
         gt_DiagSim = label_Diag_simple.tolist()
         Diag0 = Diag_Simple(pred_IDH0, pred_1p19q0, pred_CDKN0, pred_His0)
         pred_DiagSim = Diag0.tolist()

         ############################calculate tntp
         ##############IDH
         if gt_IDH[0]!=2:
             label_all_IDH.append(gt_IDH[0])
             pred_IDH_ori=F.softmax(pred_IDH_ori)
             predicted_all_IDH.append(pred_IDH_ori.detach().cpu().numpy()[0][1])
             if gt_IDH[0] == 0 and pred_IDH[0] == 0:
                 IDH_metrics['tn'] += 1
             if gt_IDH[0] == 0 and pred_IDH[0] == 1:
                 IDH_metrics['fp'] += 1
             if gt_IDH[0] == 1 and pred_IDH[0] == 0:
                 IDH_metrics['fn'] += 1
             if gt_IDH[0] == 1 and pred_IDH[0] == 1:
                 IDH_metrics['tp'] += 1
         ##############1p19q
         if gt_1p19q[0] != 2:
             label_all_1p19q.append(gt_1p19q[0])
             pred_1p19q_ori = F.softmax(pred_1p19q_ori)
             predicted_all_1p19q.append(pred_1p19q_ori.detach().cpu().numpy()[0][1])
             if gt_1p19q[0] == 0 and pred_1p19q[0] == 0:
                 p19q_metrics['tn'] += 1
             if gt_1p19q[0] == 0 and pred_1p19q[0] == 1:
                 p19q_metrics['fp'] += 1
             if gt_1p19q[0] == 1 and pred_1p19q[0] == 0:
                 p19q_metrics['fn'] += 1
             if gt_1p19q[0] == 1 and pred_1p19q[0] == 1:
                 p19q_metrics['tp'] += 1
         ##############CDKN
         if gt_CDKN[0] != 2:
             label_all_CDKN.append(gt_CDKN[0])
             pred_CDKN_ori = F.softmax(pred_CDKN_ori)
             predicted_all_CDKN.append(pred_CDKN_ori.detach().cpu().numpy()[0][1])
             if gt_CDKN[0] == 0 and pred_CDKN[0] == 0:
                 CDKN_metrics['tn'] += 1
             if gt_CDKN[0] == 0 and pred_CDKN[0] == 1:
                 CDKN_metrics['fp'] += 1
             if gt_CDKN[0] == 1 and pred_CDKN[0] == 0:
                 CDKN_metrics['fn'] += 1
             if gt_CDKN[0] == 1 and pred_CDKN[0] == 1:
                 CDKN_metrics['tp'] += 1
         ##############Diag
         label_all_Diag.append(gt_Diag[0])
         pred_Diag_ori=gene_Diag_ori(pred_IDH_ori[0],pred_1p19q_ori[0],pred_CDKN_ori[0],pred_His_ori[0],pred_Grade_ori[0])
         predicted_all_Diag.append(pred_Diag_ori)
         count_Diag+=1
         if gt_Diag[0] == pred_Diag[0]:
             correct_Diag += 1
         if gt_Diag[0] != 6:
             # G4 GBM
             if gt_Diag[0] == 0:
                 if pred_Diag[0] == 0:
                     Diag_GBM['tp'] += 1
                 else:
                     Diag_GBM['fn'] += 1
             else:
                 if not pred_Diag[0] == 0:
                     Diag_GBM['tn'] += 1
                 else:
                     Diag_GBM['fp'] += 1
             # G4 A
             if gt_Diag[0] == 1:
                 if pred_Diag[0] == 1:
                     Diag_G4A['tp'] += 1
                 else:
                     Diag_G4A['fn'] += 1
             else:
                 if not pred_Diag[0] == 1:
                     Diag_G4A['tn'] += 1
                 else:
                     Diag_G4A['fp'] += 1
             # G3 A
             if gt_Diag[0] == 2:
                 if pred_Diag[0] == 2:
                     Diag_G3A['tp'] += 1
                 else:
                     Diag_G3A['fn'] += 1
             else:
                 if not pred_Diag[0] == 2:
                     Diag_G3A['tn'] += 1
                 else:
                     Diag_G3A['fp'] += 1
             # G2 A
             if gt_Diag[0] == 3:
                 if pred_Diag[0] == 3:
                     Diag_G2A['tp'] += 1
                 else:
                     Diag_G2A['fn'] += 1
             else:
                 if not pred_Diag[0] == 3:
                     Diag_G2A['tn'] += 1
                 else:
                     Diag_G2A['fp'] += 1
             # G3 O
             if gt_Diag[0] == 4:
                 if pred_Diag[0] == 4:
                     Diag_G3O['tp'] += 1
                 else:
                     Diag_G3O['fn'] += 1
             else:
                 if not pred_Diag[0] == 4:
                     Diag_G3O['tn'] += 1
                 else:
                     Diag_G3O['fp'] += 1
             # G2 O
             if gt_Diag[0] == 5:
                 if pred_Diag[0] == 5:
                     Diag_G2O['tp'] += 1
                 else:
                     Diag_G2O['fn'] += 1
             else:
                 if not pred_Diag[0] == 5:
                     Diag_G2O['tn'] += 1
                 else:
                     Diag_G2O['fp'] += 1

         ##############DiagSim
         label_all_DiagSim.append(gt_DiagSim[0])
         pred_DiagSim_ori = gene_DiagSim_ori(pred_IDH_ori[0], pred_1p19q_ori[0], pred_CDKN_ori[0], pred_His_ori[0])
         predicted_all_DiagSim.append(pred_DiagSim_ori)
         count_Diag_Sim += 1
         if gt_DiagSim[0] == pred_DiagSim[0]:
             correct_Diag_Sim += 1
         if gt_DiagSim[0] != 4:
             # G4 GBM
             if gt_DiagSim[0] == 0:
                 if pred_DiagSim[0] == 0:
                     DiagSim_GBM['tp'] += 1
                 else:
                     DiagSim_GBM['fn'] += 1
             else:
                 if not pred_DiagSim[0] == 0:
                     DiagSim_GBM['tn'] += 1
                 else:
                     DiagSim_GBM['fp'] += 1
             # G4 A
             if gt_DiagSim[0] == 1:
                 if pred_DiagSim[0] == 1:
                     DiagSim_G4A['tp'] += 1
                 else:
                     DiagSim_G4A['fn'] += 1
             else:
                 if not pred_DiagSim[0] == 1:
                     DiagSim_G4A['tn'] += 1
                 else:
                     DiagSim_G4A['fp'] += 1
             # G23 A
             if gt_DiagSim[0] == 2:
                 if pred_DiagSim[0] == 2:
                     DiagSim_G23A['tp'] += 1
                 else:
                     DiagSim_G23A['fn'] += 1
             else:
                 if not pred_DiagSim[0] == 2:
                     DiagSim_G23A['tn'] += 1
                 else:
                     DiagSim_G23A['fp'] += 1
             # G23 O
             if gt_DiagSim[0] == 3:
                 if pred_DiagSim[0] == 3:
                     DiagSim_G23O['tp'] += 1
                 else:
                     DiagSim_G23O['fn'] += 1
             else:
                 if not pred_DiagSim[0] == 3:
                     DiagSim_G23O['tn'] += 1
                 else:
                     DiagSim_G23O['fp'] += 1

     ##########  IDH
     Acc_IDH = (IDH_metrics['tp'] + IDH_metrics['tn']) / (IDH_metrics['tp'] + IDH_metrics['tn'] + IDH_metrics['fp'] + IDH_metrics['fn'])
     IDH_metrics['sen'] = (IDH_metrics['tp']) / (IDH_metrics['tp'] + IDH_metrics['fn'] + 0.000001)  # recall
     IDH_metrics['spec'] = (IDH_metrics['tn']) / (IDH_metrics['tn'] + IDH_metrics['fp'] + 0.000001)
     IDH_metrics['pre'] = (IDH_metrics['tp']) / (IDH_metrics['tp'] + IDH_metrics['fp'] + 0.000001)
     IDH_metrics['recall'] = IDH_metrics['sen']
     IDH_metrics['f1'] = (2 * IDH_metrics['pre'] * IDH_metrics['recall']) / ( IDH_metrics['pre'] + IDH_metrics['recall'] + 0.000001)
     IDH_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_IDH), y_score=np.array(predicted_all_IDH))
     ##########  1p19q
     Acc_1p19q = (p19q_metrics['tp'] + p19q_metrics['tn']) / (p19q_metrics['tp'] + p19q_metrics['tn'] + p19q_metrics['fp'] + p19q_metrics['fn'])
     p19q_metrics['sen'] = (p19q_metrics['tp']) / (p19q_metrics['tp'] + p19q_metrics['fn'] + 0.000001)  # recall
     p19q_metrics['spec'] = (p19q_metrics['tn']) / (p19q_metrics['tn'] + p19q_metrics['fp'] + 0.000001)
     p19q_metrics['pre'] = (p19q_metrics['tp']) / (p19q_metrics['tp'] + p19q_metrics['fp'] + 0.000001)
     p19q_metrics['recall'] = p19q_metrics['sen']
     p19q_metrics['f1'] = (2 * p19q_metrics['pre'] * p19q_metrics['recall']) / (p19q_metrics['pre'] + p19q_metrics['recall'] + 0.000001)
     p19q_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_1p19q), y_score=np.array(predicted_all_1p19q))
     ##########  CDKN
     Acc_CDKN = (CDKN_metrics['tp'] + CDKN_metrics['tn']) / (CDKN_metrics['tp'] + CDKN_metrics['tn'] + CDKN_metrics['fp'] + CDKN_metrics['fn'])
     CDKN_metrics['sen'] = (CDKN_metrics['tp']) / (CDKN_metrics['tp'] + CDKN_metrics['fn'] + 0.000001)  # recall
     CDKN_metrics['spec'] = (CDKN_metrics['tn']) / (CDKN_metrics['tn'] + CDKN_metrics['fp'] + 0.000001)
     CDKN_metrics['pre'] = (CDKN_metrics['tp']) / (CDKN_metrics['tp'] + CDKN_metrics['fp'] + 0.000001)
     CDKN_metrics['recall'] = CDKN_metrics['sen']
     CDKN_metrics['f1'] = (2 * CDKN_metrics['pre'] * CDKN_metrics['recall']) / (CDKN_metrics['pre'] + CDKN_metrics['recall'] + 0.000001)
     CDKN_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_CDKN), y_score=np.array(predicted_all_CDKN))
     ##########  Diag
     Acc_Diag = correct_Diag / count_Diag
     #  Sensitivity
     Diag_GBM['sen'] = (Diag_GBM['tp']) / (Diag_GBM['tp'] + Diag_GBM['fn'] + 0.000001)
     Diag_G4A['sen'] = (Diag_G4A['tp']) / (Diag_G4A['tp'] + Diag_G4A['fn'] + 0.000001)
     Diag_G3A['sen'] = (Diag_G3A['tp']) / (Diag_G3A['tp'] + Diag_G3A['fn'] + 0.000001)
     Diag_G2A['sen'] = (Diag_G2A['tp']) / (Diag_G2A['tp'] + Diag_G2A['fn'] + 0.000001)
     Diag_G3O['sen'] = (Diag_G3O['tp']) / (Diag_G3O['tp'] + Diag_G3O['fn'] + 0.000001)
     Diag_G2O['sen'] = (Diag_G2O['tp']) / (Diag_G2O['tp'] + Diag_G2O['fn'] + 0.000001)
     Diag_all['sen'] = (Diag_GBM['sen'] + Diag_G4A['sen'] +Diag_G3A['sen']+Diag_G2A['sen'] + Diag_G3O['sen'] +Diag_G2O['sen']) / 6

     all_His_sen_weight = Diag_GBM['sen'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                          Diag_G4A['sen'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                          Diag_G3A['sen'] * label_all_Diag.count(2) / len(label_all_Diag) +\
                         Diag_G2A['sen'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                         Diag_G3O['sen'] * label_all_Diag.count(4) / len(label_all_Diag) +\
                         Diag_G2O['sen'] * label_all_Diag.count(5) / len(label_all_Diag)

         #  Spec
     Diag_GBM['spec'] = (Diag_GBM['tn']) / (Diag_GBM['tn'] + Diag_GBM['fp'] + 0.000001)
     Diag_G4A['spec'] = (Diag_G4A['tn']) / (Diag_G4A['tn'] + Diag_G4A['fp'] + 0.000001)
     Diag_G3A['spec'] = (Diag_G3A['tn']) / (Diag_G3A['tn'] + Diag_G3A['fp'] + 0.000001)
     Diag_G2A['spec'] = (Diag_G2A['tn']) / (Diag_G2A['tn'] + Diag_G2A['fp'] + 0.000001)
     Diag_G3O['spec'] = (Diag_G3O['tn']) / (Diag_G3O['tn'] + Diag_G3O['fp'] + 0.000001)
     Diag_G2O['spec'] = (Diag_G2O['tn']) / (Diag_G2O['tn'] + Diag_G2O['fp'] + 0.000001)
     Diag_all['spec'] = (Diag_GBM['spec'] + Diag_G4A['spec'] +Diag_G3A['spec']+Diag_G2A['spec'] + Diag_G3O['spec'] +Diag_G2O['spec']) / 6

     all_His_spec_weight = Diag_GBM['spec'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                          Diag_G4A['spec'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                          Diag_G3A['spec'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                          Diag_G2A['spec'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                          Diag_G3O['spec'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                          Diag_G2O['spec'] * label_all_Diag.count(5) / len(label_all_Diag)
     #  Precision
     Diag_GBM['pre'] = (Diag_GBM['tp']) / (Diag_GBM['tp'] + Diag_GBM['fp'] + 0.000001)
     Diag_G4A['pre'] = (Diag_G4A['tp']) / (Diag_G4A['tp'] + Diag_G4A['fp'] + 0.000001)
     Diag_G3A['pre'] = (Diag_G3A['tp']) / (Diag_G3A['tp'] + Diag_G3A['fp'] + 0.000001)
     Diag_G2A['pre'] = (Diag_G2A['tp']) / (Diag_G2A['tp'] + Diag_G2A['fp'] + 0.000001)
     Diag_G3O['pre'] = (Diag_G3O['tp']) / (Diag_G3O['tp'] + Diag_G3O['fp'] + 0.000001)
     Diag_G2O['pre'] = (Diag_G2O['tp']) / (Diag_G2O['tp'] + Diag_G2O['fp'] + 0.000001)
     Diag_all['pre'] = (Diag_GBM['pre'] + Diag_G4A['pre'] +Diag_G3A['pre']+Diag_G2A['pre'] + Diag_G3O['pre'] +Diag_G2O['pre']) / 6
     all_His_pre_weight = Diag_GBM['pre'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                           Diag_G4A['pre'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                           Diag_G3A['pre'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                           Diag_G2A['pre'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                           Diag_G3O['pre'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                           Diag_G2O['pre'] * label_all_Diag.count(5) / len(label_all_Diag)
     #  Recall
     Diag_GBM['recall'] = (Diag_GBM['tp']) / (Diag_GBM['tp'] + Diag_GBM['fn'] + 0.000001)
     Diag_G4A['recall'] = (Diag_G4A['tp']) / (Diag_G4A['tp'] + Diag_G4A['fn'] + 0.000001)
     Diag_G3A['recall'] = (Diag_G3A['tp']) / (Diag_G3A['tp'] + Diag_G3A['fn'] + 0.000001)
     Diag_G2A['recall'] = (Diag_G2A['tp']) / (Diag_G2A['tp'] + Diag_G2A['fn'] + 0.000001)
     Diag_G3O['recall'] = (Diag_G3O['tp']) / (Diag_G3O['tp'] + Diag_G3O['fn'] + 0.000001)
     Diag_G2O['recall'] = (Diag_G2O['tp']) / (Diag_G2O['tp'] + Diag_G2O['fn'] + 0.000001)
     Diag_all['recall'] = (Diag_GBM['recall'] + Diag_G4A['recall'] +Diag_G3A['recall']+Diag_G2A['recall'] + Diag_G3O['recall'] +Diag_G2O['recall']) / 6
     all_His_recall_weight = Diag_GBM['recall'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                           Diag_G4A['recall'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                           Diag_G3A['recall'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                           Diag_G2A['recall'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                           Diag_G3O['recall'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                           Diag_G2O['recall'] * label_all_Diag.count(5) / len(label_all_Diag)
     #  F1
     Diag_GBM['f1'] = (2 * Diag_GBM['pre'] * Diag_GBM['recall']) / (Diag_GBM['pre'] + Diag_GBM['recall'] + 0.000001)
     Diag_G4A['f1'] = (2 * Diag_G4A['pre'] * Diag_G4A['recall']) / (Diag_G4A['pre'] + Diag_G4A['recall'] + 0.000001)
     Diag_G3A['f1'] = (2 * Diag_G3A['pre'] * Diag_G3A['recall']) / (Diag_G3A['pre'] + Diag_G3A['recall'] + 0.000001)
     Diag_G2A['f1'] = (2 * Diag_G2A['pre'] * Diag_G2A['recall']) / (Diag_G2A['pre'] + Diag_G2A['recall'] + 0.000001)
     Diag_G3O['f1'] = (2 * Diag_G3O['pre'] * Diag_G3O['recall']) / (Diag_G3O['pre'] + Diag_G3O['recall'] + 0.000001)
     Diag_G2O['f1'] = (2 * Diag_G2O['pre'] * Diag_G2O['recall']) / (Diag_G2O['pre'] + Diag_G2O['recall'] + 0.000001)
     Diag_all['f1'] = (Diag_GBM['f1'] + Diag_G4A['f1'] +Diag_G3A['f1']+Diag_G2A['f1'] + Diag_G3O['f1'] +Diag_G2O['f1']) / 6
     all_His_f1_weight = Diag_GBM['f1'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                           Diag_G4A['f1'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                           Diag_G3A['f1'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                           Diag_G2A['f1'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                           Diag_G3O['f1'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                           Diag_G2O['f1'] * label_all_Diag.count(5) / len(label_all_Diag)
     # AUC
     out_cls_all_softmax_Diag = np.array(predicted_all_Diag)
     label_all_np = np.array(label_all_Diag)
     label_all_onehot = make_one_hot(label_all_np)
     fpr = dict()
     tpr = dict()
     roc_auc = dict()
     for i in range(6):
         fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Diag[:, i])
         roc_auc[i]=auc(fpr[i], tpr[i])
     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
     mean_tpr = np.zeros_like(all_fpr)
     for i in range(6):
         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
     mean_tpr /= 6
     fpr["macro"] = all_fpr
     tpr["macro"] = mean_tpr
     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
     Diag_all['AUC'] = roc_auc["macro"]
     Diag_GBM['AUC'] = roc_auc[0]
     Diag_G4A['AUC'] = roc_auc[1]
     Diag_G3A['AUC'] = roc_auc[2]
     Diag_G2A['AUC'] = roc_auc[3]
     Diag_G3O['AUC'] = roc_auc[4]
     Diag_G2O['AUC'] = roc_auc[5]
     fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Diag.ravel())
     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

     # plt.figure(dpi=600)
     # lw = 2
     # plt.plot(fpr["micro"], tpr["micro"],
     #          label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
     #          color="deeppink", linestyle=":", linewidth=4, )
     #
     # plt.plot(fpr["macro"], tpr["macro"],
     #          label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
     #          color="navy", linestyle=":", linewidth=4, )
     # from itertools import cycle
     # colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue", "green"])
     # for i, color in zip(range(6), colors):
     #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
     #              label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]), )
     #
     # plt.plot([0, 1], [0, 1], "k--", lw=lw)
     # plt.xlim([0.0, 1.0])
     # plt.ylim([0.0, 1.05])
     # plt.xlabel("False Positive Rate")
     # plt.ylabel("True Positive Rate")
     # plt.title("Receiver Operating Characteristic (ROC) curve")
     # plt.legend()
     # plt.savefig('./temp/roc.jpg')


     ##########  DiagSim
     Acc_DiagSim = correct_Diag_Sim / count_Diag_Sim
     #  Sensitivity
     DiagSim_GBM['sen'] = (DiagSim_GBM['tp']) / (DiagSim_GBM['tp'] + DiagSim_GBM['fn'] + 0.000001)
     DiagSim_G4A['sen'] = (DiagSim_G4A['tp']) / (DiagSim_G4A['tp'] + DiagSim_G4A['fn'] + 0.000001)
     DiagSim_G23A['sen'] = (DiagSim_G23A['tp']) / (DiagSim_G23A['tp'] + DiagSim_G23A['fn'] + 0.000001)
     DiagSim_G23O['sen'] = (DiagSim_G23O['tp']) / (DiagSim_G23O['tp'] + DiagSim_G23O['fn'] + 0.000001)
     DiagSim_all['sen'] = (DiagSim_GBM['sen'] + DiagSim_G4A['sen'] + DiagSim_G23A['sen'] + DiagSim_G23O['sen']) / 4
     #  Spec
     DiagSim_GBM['spec'] = (DiagSim_GBM['tn']) / (DiagSim_GBM['tn'] + DiagSim_GBM['fp'] + 0.000001)
     DiagSim_G4A['spec'] = (DiagSim_G4A['tn']) / (DiagSim_G4A['tn'] + DiagSim_G4A['fp'] + 0.000001)
     DiagSim_G23A['spec'] = (DiagSim_G23A['tn']) / (DiagSim_G23A['tn'] + DiagSim_G23A['fp'] + 0.000001)
     DiagSim_G23O['spec'] = (DiagSim_G23O['tn']) / (DiagSim_G23O['tn'] + DiagSim_G23O['fp'] + 0.000001)
     DiagSim_all['spec'] = (DiagSim_GBM['spec'] + DiagSim_G4A['spec'] + DiagSim_G23A['spec'] + DiagSim_G23O['spec']) / 4
     #  Precision
     DiagSim_GBM['pre'] = (DiagSim_GBM['tp']) / (DiagSim_GBM['tp'] + DiagSim_GBM['fp'] + 0.000001)
     DiagSim_G4A['pre'] = (DiagSim_G4A['tp']) / (DiagSim_G4A['tp'] + DiagSim_G4A['fp'] + 0.000001)
     DiagSim_G23A['pre'] = (DiagSim_G23A['tp']) / (DiagSim_G23A['tp'] + DiagSim_G23A['fp'] + 0.000001)
     DiagSim_G23O['pre'] = (DiagSim_G23O['tp']) / (DiagSim_G23O['tp'] + DiagSim_G23O['fp'] + 0.000001)
     DiagSim_all['pre'] = (DiagSim_GBM['pre'] + DiagSim_G4A['pre'] + DiagSim_G23A['pre'] + DiagSim_G23O['pre']) / 4
     #  Recall
     DiagSim_GBM['recall'] = (DiagSim_GBM['tp']) / (DiagSim_GBM['tp'] + DiagSim_GBM['fn'] + 0.000001)
     DiagSim_G4A['recall'] = (DiagSim_G4A['tp']) / (DiagSim_G4A['tp'] + DiagSim_G4A['fn'] + 0.000001)
     DiagSim_G23A['recall'] = (DiagSim_G23A['tp']) / (DiagSim_G23A['tp'] + DiagSim_G23A['fn'] + 0.000001)
     DiagSim_G23O['recall'] = (DiagSim_G23O['tp']) / (DiagSim_G23O['tp'] + DiagSim_G23O['fn'] + 0.000001)
     DiagSim_all['recall'] = (DiagSim_GBM['recall'] + DiagSim_G4A['recall'] + DiagSim_G23A['recall'] + DiagSim_G23O[
         'recall']) / 4
     #  F1
     DiagSim_GBM['f1'] = (2 * DiagSim_GBM['pre'] * DiagSim_GBM['recall']) / (DiagSim_GBM['pre'] + DiagSim_GBM['recall'] + 0.000001)
     DiagSim_G4A['f1'] = (2 * DiagSim_G4A['pre'] * DiagSim_G4A['recall']) / (DiagSim_G4A['pre'] + DiagSim_G4A['recall'] + 0.000001)
     DiagSim_G23A['f1'] = (2 * DiagSim_G23A['pre'] * DiagSim_G23A['recall']) / (DiagSim_G23A['pre'] + DiagSim_G23A['recall'] + 0.000001)
     DiagSim_G23O['f1'] = (2 * DiagSim_G23O['pre'] * DiagSim_G23O['recall']) / (DiagSim_G23O['pre'] + DiagSim_G23O['recall'] + 0.000001)
     DiagSim_all['f1'] = (DiagSim_GBM['f1'] + DiagSim_G4A['f1'] + DiagSim_G23A['f1'] + DiagSim_G23O['f1']) / 4
     # AUC
     out_cls_all_softmax_DiagSim = np.array(predicted_all_DiagSim)
     label_all_np = np.array(label_all_DiagSim)
     label_all_onehot = make_one_hot(label_all_np)
     fpr = dict()
     tpr = dict()
     roc_auc = dict()
     for i in range(4):
         fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_DiagSim[:, i])
     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
     mean_tpr = np.zeros_like(all_fpr)
     for i in range(4):
         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
     mean_tpr /= 4
     fpr["macro"] = all_fpr
     tpr["macro"] = mean_tpr
     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
     DiagSim_all['AUC'] = roc_auc["macro"]


     list_IDH = [Acc_IDH, IDH_metrics['sen'], IDH_metrics['spec'], IDH_metrics['pre'], IDH_metrics['recall'], IDH_metrics['f1'], IDH_metrics['AUC']]
     list_1p19q = [Acc_1p19q, p19q_metrics['sen'], p19q_metrics['spec'], p19q_metrics['pre'], p19q_metrics['recall'], p19q_metrics['f1'], p19q_metrics['AUC']]
     list_CDKN = [Acc_CDKN, CDKN_metrics['sen'], CDKN_metrics['spec'], CDKN_metrics['pre'], CDKN_metrics['recall'], CDKN_metrics['f1'], CDKN_metrics['AUC']]
     list_Diag = [Acc_Diag, Diag_all['sen'], Diag_all['spec'], Diag_all['pre'], Diag_all['recall'], Diag_all['f1'], Diag_all['AUC']]
     list_DiagSim = [Acc_DiagSim, DiagSim_all['sen'], DiagSim_all['spec'], DiagSim_all['pre'], DiagSim_all['recall'], DiagSim_all['f1'], DiagSim_all['AUC']]


     return list_IDH, list_1p19q, list_CDKN,list_Diag,list_DiagSim


def gene_Diag_ori(pred_IDH_ori,pred_1p19q_ori,pred_CDKN_ori,pred_His_ori,pred_Grade_ori):
    pred_IDH_ori=pred_IDH_ori.detach().cpu().numpy()
    pred_1p19q_ori = pred_1p19q_ori.detach().cpu().numpy()
    pred_CDKN_ori = pred_CDKN_ori.detach().cpu().numpy()
    pred_His_ori = pred_His_ori.detach().cpu().numpy()
    pred_Grade_ori = pred_Grade_ori.detach().cpu().numpy()
    G4GBM_prob=pred_IDH_ori[0]
    G3O_prob=pred_IDH_ori[1]*pred_1p19q_ori[1]*(1-pred_Grade_ori[0])
    G2O_prob = pred_IDH_ori[1] * pred_1p19q_ori[1] * pred_Grade_ori[0]
    G4A_prob =pred_IDH_ori[1] * pred_1p19q_ori[0]*(1-pred_CDKN_ori[0]*(1-pred_His_ori[2]))
    G3A_prob = pred_IDH_ori[1] * pred_1p19q_ori[0]*pred_CDKN_ori[0]*(1-pred_His_ori[2])*(1-pred_Grade_ori[0])
    G2A_prob = pred_IDH_ori[1] * pred_1p19q_ori[0] * pred_CDKN_ori[0] * (1 - pred_His_ori[2]) * pred_Grade_ori[0]
    pred_Diag_ori=[G4GBM_prob,G4A_prob,G3A_prob,G2A_prob,G3O_prob,G2O_prob]
    pred_Diag_ori=np.array(pred_Diag_ori)
    return pred_Diag_ori

def gene_DiagSim_ori(pred_IDH_ori,pred_1p19q_ori,pred_CDKN_ori,pred_His_ori):
    pred_IDH_ori=pred_IDH_ori.detach().cpu().numpy()
    pred_1p19q_ori = pred_1p19q_ori.detach().cpu().numpy()
    pred_CDKN_ori = pred_CDKN_ori.detach().cpu().numpy()
    pred_His_ori = pred_His_ori.detach().cpu().numpy()
    G4GBM_prob=pred_IDH_ori[0]
    G23O_prob=pred_IDH_ori[1]*pred_1p19q_ori[1]
    G4A_prob =pred_IDH_ori[1] * pred_1p19q_ori[0]*(1-pred_CDKN_ori[0]*(1-pred_His_ori[2]))
    G23A_prob = pred_IDH_ori[1] * pred_1p19q_ori[0]*pred_CDKN_ori[0]*(1-pred_His_ori[2])
    pred_Diag_ori=[G4GBM_prob,G4A_prob,G23A_prob,G23O_prob]
    pred_Diag_ori=np.array(pred_Diag_ori)
    return pred_Diag_ori

from numpy import dot
from numpy.linalg import norm
def feature_space_similarity(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, dataloader, gpuID):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    if not os.path.exists('./noise/'):
        os.makedirs('./noise/')
    # else:
    #     remove_all_file('./noise/')

    test_bar = tqdm(dataloader)
    count = 0
    count_noiseclean_His=0
    count_noiseclean_Grade = 0
    His_clean_features_A = []
    His_clean_features_O = []
    His_clean_features_GBM = []
    His_noise_features_A= {}
    His_noise_features_O = {}
    His_noise_features_GBM = {}
    Grade_clean_features_G2 = []
    Grade_clean_features_G3 = []
    Grade_clean_features_G4 = []
    Grade_noise_features_G2 = {}
    Grade_noise_features_G3 = {}
    Grade_noise_features_G4 = {}
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name=packs[2]
        cleannoise_num=packs[3]
        count += 1
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]
        imp_his, imp_grade = imp_gene(opt,img)
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature, imp_his,
                                                                                            imp_grade)
        results_dict,saliency_A,saliency_O,saliency_GBM,saliency_G2,saliency_G3,saliency_G4,feature_His,feature_Grade = Mine_model_Cls(encoded_His, encoded_Grade)

        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        gt_His = label_his.tolist() #[BS] A  O GBM//0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        gt_Grade = label_grade.tolist()  # [BS] A  O GBM//0 1 2

        ################################ FEATURE SIMILARITY ################################
        # clean_num=cleannoise_num.numpy()[0][0]
        # if count_noiseclean_His<clean_num:
        #     if gt_His[0] == 0:
        #         feature_His = F.softmax(feature_His, dim=1).detach().cpu().numpy()[0]
        #         His_clean_features_A.append(feature_His)
        #     if gt_His[0] == 1:
        #         feature_His = F.softmax(feature_His, dim=1).detach().cpu().numpy()[0]
        #         His_clean_features_O.append(feature_His)
        #     if gt_His[0] == 2:
        #         feature_His = F.softmax(feature_His, dim=1).detach().cpu().numpy()[0]
        #         His_clean_features_GBM.append(feature_His)
        #     count_noiseclean_His += 1
        # else:
        #     if gt_His[0]==0:
        #         feature_His = F.softmax(feature_His, dim=1).detach().cpu().numpy()[0]
        #         His_noise_features_A[file_name[0]]=feature_His
        #     elif gt_His[0]==1:
        #         feature_His = F.softmax(feature_His, dim=1).detach().cpu().numpy()[0]
        #         His_noise_features_O[file_name[0]]=feature_His
        #     elif gt_His[0] == 2:
        #         feature_His = F.softmax(feature_His, dim=1).detach().cpu().numpy()[0]
        #         His_noise_features_GBM[file_name[0]]=feature_His

        clean_num = cleannoise_num.numpy()[0][0]
        if count_noiseclean_Grade < clean_num:
            if gt_Grade[0] == 0:
                feature_Grade = F.softmax(feature_Grade, dim=1).detach().cpu().numpy()[0]
                Grade_clean_features_G2.append(feature_Grade)
            if gt_Grade[0] == 1:
                feature_Grade = F.softmax(feature_Grade, dim=1).detach().cpu().numpy()[0]
                Grade_clean_features_G3.append(feature_Grade)
            if gt_Grade[0] == 2:
                feature_Grade = F.softmax(feature_Grade, dim=1).detach().cpu().numpy()[0]
                Grade_clean_features_G4.append(feature_Grade)
            count_noiseclean_Grade += 1
        else:
            if gt_Grade[0] == 0:
                feature_Grade = F.softmax(feature_Grade, dim=1).detach().cpu().numpy()[0]
                Grade_noise_features_G2[file_name[0]] = feature_Grade
            elif gt_Grade[0] == 1:
                feature_Grade = F.softmax(feature_Grade, dim=1).detach().cpu().numpy()[0]
                Grade_noise_features_G3[file_name[0]] = feature_Grade
            elif gt_Grade[0] == 2:
                feature_Grade = F.softmax(feature_Grade, dim=1).detach().cpu().numpy()[0]
                Grade_noise_features_G4[file_name[0]] = feature_Grade
        ################################ FEATURE SIMILARITY ################################


    # with open('noise/His_Train_noise.txt', "w+") as f:
    #     cosine_similarity ={}
    #     for file_name, feature_noise in His_noise_features_A.items():
    #         cosine_similarity[file_name]=0
    #         for i in range(len(His_clean_features_A)):
    #             cosine_similarity[file_name]+=dot(feature_noise, His_clean_features_A[i])/(norm(feature_noise)*norm(His_clean_features_A[i]))
    #         cosine_similarity[file_name]=cosine_similarity[file_name]/len(His_clean_features_A)
    #     cosine_similarity = sorted(cosine_similarity,reverse =True)
    #     for i in range(int(len(cosine_similarity)*0.9)):
    #         f.write(cosine_similarity[i])
    #         f.write('\n')
    #         f.flush()
    #     cosine_similarity = {}
    #     for file_name, feature_noise in His_noise_features_O.items():
    #         cosine_similarity[file_name] = 0
    #         for i in range(len(His_clean_features_O)):
    #             cosine_similarity[file_name] += dot(feature_noise, His_clean_features_O[i]) / (
    #                         norm(feature_noise) * norm(His_clean_features_O[i]))
    #         cosine_similarity[file_name] = cosine_similarity[file_name] / len(His_clean_features_O)
    #     cosine_similarity = sorted(cosine_similarity, reverse=True)
    #     for i in range(int(len(cosine_similarity) * 0.9)):
    #         f.write(cosine_similarity[i])
    #         f.write('\n')
    #         f.flush()
    #     cosine_similarity = {}
    #     for file_name, feature_noise in His_noise_features_GBM.items():
    #         cosine_similarity[file_name] = 0
    #         for i in range(len(His_clean_features_GBM)):
    #             cosine_similarity[file_name] += dot(feature_noise, His_clean_features_GBM[i]) / (
    #                         norm(feature_noise) * norm(His_clean_features_GBM[i]))
    #         cosine_similarity[file_name] = cosine_similarity[file_name] / len(His_clean_features_GBM)
    #     cosine_similarity = sorted(cosine_similarity, reverse=True)
    #     for i in range(int(len(cosine_similarity) * 0.9)):
    #         f.write(cosine_similarity[i])
    #         f.write('\n')
    #         f.flush()



    with open('noise/Grade_Train_noise.txt', "w+") as f:
        cosine_similarity ={}
        for file_name, feature_noise in Grade_noise_features_G2.items():
            cosine_similarity[file_name]=0
            for i in range(len(Grade_clean_features_G2)):
                cosine_similarity[file_name]+=dot(feature_noise, Grade_clean_features_G2[i])/(norm(feature_noise)*norm(Grade_clean_features_G2[i]))
            cosine_similarity[file_name]=cosine_similarity[file_name]/len(Grade_clean_features_G2)
        cosine_similarity = sorted(cosine_similarity,reverse =True)
        for i in range(int(len(cosine_similarity)*0.9)):
            f.write(cosine_similarity[i])
            f.write('\n')
            f.flush()
        cosine_similarity = {}
        for file_name, feature_noise in Grade_noise_features_G3.items():
            cosine_similarity[file_name] = 0
            for i in range(len(Grade_clean_features_G3)):
                cosine_similarity[file_name] += dot(feature_noise, Grade_clean_features_G3[i]) / (
                            norm(feature_noise) * norm(Grade_clean_features_G3[i]))
            cosine_similarity[file_name] = cosine_similarity[file_name] / len(Grade_clean_features_G3)
        cosine_similarity = sorted(cosine_similarity, reverse=True)
        for i in range(int(len(cosine_similarity) * 0.9)):
            f.write(cosine_similarity[i])
            f.write('\n')
            f.flush()
        cosine_similarity = {}
        for file_name, feature_noise in Grade_noise_features_G4.items():
            cosine_similarity[file_name] = 0
            for i in range(len(Grade_clean_features_G4)):
                cosine_similarity[file_name] += dot(feature_noise, Grade_clean_features_G4[i]) / (
                            norm(feature_noise) * norm(Grade_clean_features_G4[i]))
            cosine_similarity[file_name] = cosine_similarity[file_name] / len(Grade_clean_features_G4)
        cosine_similarity = sorted(cosine_similarity, reverse=True)
        for i in range(int(len(cosine_similarity) * 0.9)):
            f.write(cosine_similarity[i])
            f.write('\n')
            f.flush()
    a=1




def validation_His(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    # resnet.eval()
    if 1:

        count_His = 0
        count_His_NoOA = 0
        correct_His = 0
        correct_His2=0
        correct_His3 = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        AO_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}

        label_all_His = []
        predicted_all_His = []

    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_His = label[:, 3]
        if opt['name'].split('_')[0]=='CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits'][3]

        _, pred_His = torch.max(pred_ori.data, 1)
        _, pred_His_2 = torch.max(pred_ori[:,1:].data, 1)
        pred_His = pred_His.tolist()  # [BS] AO A  O GBM //0 1 2 3
        pred_His_2 = pred_His_2.tolist()  #[BS] A  O GBM //0 1 2
        gt_His = label_His.tolist() #[BS] AO A  O GBM//0 1 2 3


        for j in range(bs):
            ##################   His
            # AO
            # if pred_His[j] == 0:
            #     if gt_His[j] == 0 :
            #         AO_metrics['tn'] += 1
            #     else:
            #         AO_metrics['fn'] += 1
            # else:
            #     if not gt_His[j] == 0:
            #         AO_metrics['tp'] += 1
            #     else:
            #         AO_metrics['fp'] += 1
            if not gt_His[j] == 0:
                # A
                if pred_His_2[j] == 0:
                    if gt_His[j] == 1:
                        A_metrics['tn'] += 1
                    else:
                        A_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 1:
                        A_metrics['tp'] += 1
                    else:
                        A_metrics['fp'] += 1
                # O
                if pred_His_2[j] == 1:
                    if gt_His[j] == 2:
                        O_metrics['tn'] += 1
                    else:
                        O_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 2:
                        O_metrics['tp'] += 1
                    else:
                        O_metrics['fp'] += 1
                # GBM
                if pred_His_2[j] == 2:
                    if gt_His[j] == 3:
                        GBM_metrics['tn'] += 1
                    else:
                        GBM_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 3:
                        GBM_metrics['tp'] += 1
                    else:
                        GBM_metrics['fp'] += 1

            gt_cm_label_His = gt_His[j]
            pred_cm_label_His = pred_His[j]
            cm_y_His = np.append(cm_y_His, gt_cm_label_His)
            cm_pred_His = np.append(cm_pred_His, pred_cm_label_His)
            label_all_His.append(gt_His[j])
            predicted_all_His.append(pred_ori.detach().cpu().numpy()[j])
            count_His += 1

            if gt_His[j] == pred_His[j]:
                correct_His2 += 1

            if gt_His[j] == 0 and (pred_His_2[j] == 0 or pred_His_2[j] == 1):
                correct_His+=1
            if gt_His[j] == 1 and pred_His_2[j]==0:
                correct_His += 1
            if gt_His[j] == 2 and pred_His_2[j]==1:
                correct_His += 1
            if gt_His[j] == 3 and pred_His_2[j]==2:
                correct_His += 1


            if not gt_His[j] == 0:
                count_His_NoOA+= 1
            if gt_His[j] == 1 and pred_His_2[j]==0:
                correct_His3 += 1
            if gt_His[j] == 2 and pred_His_2[j]==1:
                correct_His3 += 1
            if gt_His[j] == 3 and pred_His_2[j]==2:
                correct_His3 += 1


    ################################################   His
    Acc_His = correct_His / count_His
    Acc_His2 = correct_His2 / count_His
    Acc_His3 = correct_His3 / count_His_NoOA
    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    # AO_metrics['sen'] = (AO_metrics['tp']) / (AO_metrics['tp'] + AO_metrics['fn']+0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics['sen'] = (A_metrics['sen'] + AO_metrics['sen'] + O_metrics['sen'] +
                          GBM_metrics['sen'] ) / 3
    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp']+0.000001)
    # AO_metrics['spec'] = (AO_metrics['tn']) / (AO_metrics['tn'] + AO_metrics['fp']+0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp']+0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp']+0.000001)
    all_metrics['spec'] = (A_metrics['spec'] + AO_metrics['spec'] + O_metrics['spec'] +
                           GBM_metrics['spec'] ) / 3
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp']+0.000001)
    # AO_metrics['pre'] = (AO_metrics['tp']) / (AO_metrics['tp'] + AO_metrics['fp']+0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp']+0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp']+0.000001)
    all_metrics['pre'] = (A_metrics['pre'] + AO_metrics['pre'] + O_metrics['pre'] +
                          GBM_metrics['pre'] ) / 3
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    # AO_metrics['recall'] = (AO_metrics['tp']) / (AO_metrics['tp'] + AO_metrics['fn']+0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics['recall'] = (A_metrics['recall'] + AO_metrics['recall'] + O_metrics['recall'] +
                             GBM_metrics['recall'] ) / 3
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
                A_metrics['pre'] + A_metrics['recall']+0.000001)
    # AO_metrics['f1'] = (2 * AO_metrics['pre'] * AO_metrics['recall']) / (
    #             AO_metrics['pre'] + AO_metrics['recall']+0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
                O_metrics['pre'] + O_metrics['recall']+0.000001)
    GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (GBM_metrics['pre'] + GBM_metrics['recall']+0.000001)
    all_metrics['f1'] = (A_metrics['f1'] + AO_metrics['f1'] + O_metrics['f1'] +
                          GBM_metrics['f1']) / 3
    # AUC
    # out_cls_all_softmax = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
    # label_all_np = np.array(label_all_His)
    # label_all_onehot = make_one_hot(label_all_np)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(4):
    #     fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax[:, i])
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(4):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # mean_tpr /= 4
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # all_metrics['AUC'] = roc_auc["macro"]

    list_His = ( 0,None, all_metrics['f1'], all_metrics['sen'], all_metrics['spec'], 0 ,
                 all_metrics['pre'],Acc_His3)

    return list_His


def validation_Diag(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    # resnet.eval()
    if 1:

        count_Diag = 0
        correct_Diag = 0
        G23_O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                         'AUC': 0}
        G23_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                         'AUC': 0}
        G4_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}

        label_all_Diag = []
        predicted_all_Diag = []
        pred_all_Diag = []
        cm_y_Diag = []
        cm_pred_Diag = []
    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    for packs in test_bar:
        img = packs[0][0]
        label = packs[1]
        imgPath = packs[2]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        # img = resnet(img)
        if opt['name'].split('_')[0]=='CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits']

        _, pred_Diag = torch.max(pred_ori.data, 1)
        pred_Diag = pred_Diag.tolist()
        gt_Diag = label.tolist()


        for j in range(bs):
            ##################   Diag
            # GBM
            if pred_Diag[j] == 0:
                if gt_Diag[j] == 0:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 0:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
            # G4_A
            if pred_Diag[j] == 1:
                if gt_Diag[j] == 1:
                    G4_A_metrics['tp'] += 1
                else:
                    G4_A_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 1:
                    G4_A_metrics['tn'] += 1
                else:
                    G4_A_metrics['fp'] += 1
            # G23_A
            if pred_Diag[j] == 2:
                if gt_Diag[j] == 2:
                    G23_A_metrics['tp'] += 1
                else:
                    G23_A_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 2:
                    G23_A_metrics['tn'] += 1
                else:
                    G23_A_metrics['fp'] += 1
            # G23_O
            if pred_Diag[j] == 3:
                if gt_Diag[j] == 3:
                    G23_O_metrics['tn'] += 1
                else:
                    G23_O_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 3:
                    G23_O_metrics['tp'] += 1
                else:
                    G23_O_metrics['fp'] += 1

            gt_cm_label_Diag = gt_Diag[j]
            pred_cm_label_Diag = pred_Diag[j]
            cm_y_Diag = np.append(cm_y_Diag, gt_cm_label_Diag)
            cm_pred_Diag = np.append(cm_pred_Diag, pred_cm_label_Diag)
            label_all_Diag.append(gt_Diag[j])
            predicted_all_Diag.append(pred_ori.detach().cpu().numpy()[j])
            count_Diag += 1
            if gt_Diag[j] == pred_Diag[j]:
                correct_Diag += 1


    ################################################   Diag
    Acc_Diag = correct_Diag / count_Diag
    #  Sensitivity
    G23_O_metrics['sen'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fn'] + 0.000001)
    G23_A_metrics['sen'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fn'] + 0.000001)
    G4_A_metrics['sen'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fn'] + 0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics['sen'] = (G23_O_metrics['sen'] + G23_A_metrics['sen'] + G4_A_metrics['sen'] + GBM_metrics['sen']) / 4
    #  Spec
    G23_O_metrics['spec'] = (G23_O_metrics['tn']) / (G23_O_metrics['tn'] + G23_O_metrics['fp'] + 0.000001)
    G23_A_metrics['spec'] = (G23_A_metrics['tn']) / (G23_A_metrics['tn'] + G23_A_metrics['fp'] + 0.000001)
    G4_A_metrics['spec'] = (G4_A_metrics['tn']) / (G4_A_metrics['tn'] + G4_A_metrics['fp'] + 0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp'] + 0.000001)
    all_metrics['spec'] = (G23_O_metrics['spec'] + G23_A_metrics['spec'] + G4_A_metrics['spec'] + GBM_metrics[
        'spec']) / 4
    #  Precision
    G23_O_metrics['pre'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fp'] + 0.000001)
    G23_A_metrics['pre'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fp'] + 0.000001)
    G4_A_metrics['pre'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fp'] + 0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp'] + 0.000001)
    all_metrics['pre'] = (G23_O_metrics['pre'] + G23_A_metrics['pre'] + G4_A_metrics['pre'] + GBM_metrics['pre']) / 4
    #  Recall
    G23_O_metrics['recall'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fn'] + 0.000001)
    G23_A_metrics['recall'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fn'] + 0.000001)
    G4_A_metrics['recall'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fn'] + 0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics['recall'] = (G23_O_metrics['recall'] + G23_A_metrics['recall'] + G4_A_metrics['recall'] + GBM_metrics[
        'recall']) / 4

    #  F1
    G23_O_metrics['f1'] = 2 * (G23_O_metrics['pre'] * G23_O_metrics['recall']) / (
                G23_O_metrics['pre'] + G23_O_metrics['recall'] + 0.000001)
    G23_A_metrics['f1'] = 2 * (G23_A_metrics['pre'] * G23_A_metrics['recall']) / (
                G23_A_metrics['pre'] + G23_A_metrics['recall'] + 0.000001)
    G4_A_metrics['f1'] = 2 * (G4_A_metrics['pre'] * G4_A_metrics['recall']) / (
                G4_A_metrics['pre'] + G4_A_metrics['recall'] + 0.000001)
    GBM_metrics['f1'] = 2 * (GBM_metrics['pre'] * GBM_metrics['recall']) / (
                GBM_metrics['pre'] + GBM_metrics['recall'] + 0.000001)
    all_metrics['f1'] = (G23_O_metrics['f1'] + G23_A_metrics['f1'] + G4_A_metrics['f1'] + GBM_metrics['f1']) / 4
    # AUC

    # out_cls_all_softmax = F.softmax(torch.from_numpy(np.array(predicted_all_Diag)), dim=1).numpy()
    # label_all_np = np.array(label_all_Diag)
    # label_all_onehot = make_one_hot(label_all_np)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(6):
    #     fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax[:, i])
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(6):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # # Finally average it and compute AUC
    # mean_tpr /= 6
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # all_metrics['AUC'] = roc_auc["macro"]
    if eva_cm:
        cm_Diag = confusion_matrix(cm_y_Diag, cm_pred_Diag)
    else:
        cm_Diag = None
    list_Diag = (Acc_Diag, cm_Diag, all_metrics['f1'], all_metrics['sen'], all_metrics['spec'], all_metrics['AUC'],
                 all_metrics['pre'])

    return list_Diag


def make_one_hot(data1,N=0):
    if N!=0:
        num=N
    else:
        num = int(np.max(data1) + 1)
    return (np.arange(num)==data1[:,None]).astype(np.int16)

import numpy as np

def calculate_sensitivity_specificity(pred, gt):
    """
    Calculate sensitivity and specificity.

    Args:
    pred (np.ndarray): 1D array of predictions (binary, 0 or 1).
    gt (np.ndarray): 1D array of ground truth (binary, 0 or 1).

    Returns:
    tuple: (sensitivity, specificity)
    """
    # Ensure inputs are NumPy arrays
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    # Compute TP, TN, FP, FN
    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    # Calculate sensitivity and specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return sensitivity, specificity


def calculate_tp():
    a=1

import torch

def Diag_process(label,pred_IDH,pred_1p19q,pred_CDKN,pred_His_2class):
    correct_Diag=0

    label_Diag = label[:, 5].tolist()#(BS)


    _, pred_IDH = torch.max(pred_IDH.data, 1)
    _, pred_1p19q = torch.max(pred_1p19q.data, 1)
    _, pred_CDKN = torch.max(pred_CDKN.data, 1)
    _, pred_His_2class = torch.max(pred_His_2class.data, 1)

    pred_IDH = pred_IDH.tolist()
    pred_1p19q = pred_1p19q.tolist()
    pred_CDKN = pred_CDKN.tolist()
    pred_His_2class = pred_His_2class.tolist()

    """
    label 2021={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G4_A', 5:'GBM'}
    label 2021={ 0:'GBM', 1:'G4_A', 2:'G2/3_A', 3:'G2/3_O'}
    """
    for j in range(label.detach().cpu().numpy().shape[0]):
        if pred_IDH[j]==0:
            if label_Diag[j]==0:
                correct_Diag+=1
        if pred_IDH[j] == 1 and pred_1p19q[j] == 1:
            if label_Diag[j]==3:
                correct_Diag += 1
        if pred_IDH[j] == 1 and pred_1p19q[j] == 0:
            if pred_CDKN[j]==1 or pred_His_2class[j]==1:
                if label_Diag[j]==1:
                    correct_Diag += 1
            else :
                if label_Diag[j] == 2:
                    correct_Diag += 1



    return correct_Diag

def Diag_predict(pred_IDH,pred_1p19q,pred_CDKN,pred_His_2class):
    _, pred_IDH = torch.max(pred_IDH.data, 1)
    _, pred_1p19q = torch.max(pred_1p19q.data, 1)
    _, pred_CDKN = torch.max(pred_CDKN.data, 1)
    # _, pred_His = torch.max(pred_His.data, 1)
    _, pred_His_2class = torch.max(pred_His_2class.data, 1)

    pred_IDH = pred_IDH.tolist()
    pred_1p19q = pred_1p19q.tolist()
    pred_CDKN = pred_CDKN.tolist()
    # pred_His = pred_His.tolist()
    pred_His_2class = pred_His_2class.tolist()

    pred_Diag=[]

    for j in range(len(pred_IDH)):
        if pred_IDH[j]==0:
            pred_Diag.append(0)
        if pred_IDH[j] == 1 and pred_1p19q[j] == 1:
            pred_Diag.append(3)
        if pred_IDH[j] == 1 and pred_1p19q[j] == 0:
            if pred_CDKN[j]==1 or pred_His_2class[j]==1:
                pred_Diag.append(1)
            else :
                pred_Diag.append(2)
    return pred_Diag


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label, transform='softmax'):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        if transform == 'softmax':
            probs = F.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
        alpha = self.alpha * lb_one_hot + (1 - self.alpha) * (1 - lb_one_hot)
        loss = -alpha * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum() / n_valid
        return loss

from model import *
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     # if seed == 0:
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def setup_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"




def dataset_merge_3fold(opt,phase):
    np.random.seed(opt['seed'])
    random.seed(opt['seed'])
    surv = True if opt['name'].split('_')[0] == 'surv' else False
    #### GBMatch
    excel_label_wsi_GBMatch = pd.read_excel(opt['label_path'] + 'GBMatch.xlsx', sheet_name='Sheet1', header=0)
    excel_wsi_GBMatch = excel_label_wsi_GBMatch.values
    PATIENT_LIST_GBMatch = excel_wsi_GBMatch[:, 0]
    PATIENT_LIST_GBMatch = np.unique(PATIENT_LIST_GBMatch)
    np.random.shuffle(PATIENT_LIST_GBMatch)
    NUM_PATIENT_ALL = len(PATIENT_LIST_GBMatch)
    if opt['fold'] == 0:
        TRAIN_PATIENT_LIST_GBMatch = PATIENT_LIST_GBMatch[0:int(NUM_PATIENT_ALL * 0.67)]
        TEST_PATIENT_LIST_GBMatch = PATIENT_LIST_GBMatch[int(NUM_PATIENT_ALL * 0.67):]
    elif opt['fold'] == 1:
        TRAIN_PATIENT_LIST_GBMatch = np.concatenate((PATIENT_LIST_GBMatch[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST_GBMatch[int(NUM_PATIENT_ALL * 0.67):]))
        TEST_PATIENT_LIST_GBMatch = PATIENT_LIST_GBMatch[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
    elif opt['fold'] == 2:
        TRAIN_PATIENT_LIST_GBMatch = PATIENT_LIST_GBMatch[int(NUM_PATIENT_ALL * 0.33):]
        TEST_PATIENT_LIST_GBMatch = PATIENT_LIST_GBMatch[0:int(NUM_PATIENT_ALL * 0.33)]

    TRAIN_LIST_GBMatch = []
    TEST_LIST_GBMatch = []
    for i in range(excel_wsi_GBMatch.shape[0]):  # 2612
        if surv:
            if excel_wsi_GBMatch[:, 0][i] in TRAIN_PATIENT_LIST_GBMatch  and (isinstance(excel_wsi_GBMatch[i, 14], int) or isinstance(excel_wsi_GBMatch[i, 14], float)) and not np.isnan(excel_wsi_GBMatch[i, 14]):
                TRAIN_LIST_GBMatch.append(excel_wsi_GBMatch[i, :])
            elif excel_wsi_GBMatch[:, 0][i] in TEST_PATIENT_LIST_GBMatch and (isinstance(excel_wsi_GBMatch[i, 14], int) or isinstance(excel_wsi_GBMatch[i, 14], float)) and not np.isnan(excel_wsi_GBMatch[i, 14]):
                TEST_LIST_GBMatch.append(excel_wsi_GBMatch[i, :])
        else:
            if excel_wsi_GBMatch[:, 0][i] in TRAIN_PATIENT_LIST_GBMatch:
                TRAIN_LIST_GBMatch.append(excel_wsi_GBMatch[i, :])
            elif excel_wsi_GBMatch[:, 0][i] in TEST_PATIENT_LIST_GBMatch:
                TEST_LIST_GBMatch.append(excel_wsi_GBMatch[i, :])
    LIST_GBMatch = TRAIN_LIST_GBMatch if phase == 'Train' else TEST_LIST_GBMatch

    #### IvYGAP
    excel_label_wsi_IvYGAP = pd.read_excel(opt['label_path'] + 'IvYGAP.xlsx', sheet_name='Sheet1', header=0)
    excel_wsi_IvYGAP = excel_label_wsi_IvYGAP.values
    PATIENT_LIST_IvYGAP = excel_wsi_IvYGAP[:, 0]
    PATIENT_LIST_IvYGAP = np.unique(PATIENT_LIST_IvYGAP)
    np.random.shuffle(PATIENT_LIST_IvYGAP)

    NUM_PATIENT_ALL = len(PATIENT_LIST_IvYGAP)
    if opt['fold'] == 0:
        TRAIN_PATIENT_LIST_IvYGAP = PATIENT_LIST_IvYGAP[0:int(NUM_PATIENT_ALL * 0.67)]
        TEST_PATIENT_LIST_IvYGAP = PATIENT_LIST_IvYGAP[int(NUM_PATIENT_ALL * 0.67):]
    elif opt['fold'] == 1:
        TRAIN_PATIENT_LIST_IvYGAP = np.concatenate(
            (PATIENT_LIST_IvYGAP[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST_IvYGAP[int(NUM_PATIENT_ALL * 0.67):]))
        TEST_PATIENT_LIST_IvYGAP = PATIENT_LIST_IvYGAP[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
    elif opt['fold'] == 2:
        TRAIN_PATIENT_LIST_IvYGAP = PATIENT_LIST_IvYGAP[int(NUM_PATIENT_ALL * 0.33):]
        TEST_PATIENT_LIST_IvYGAP = PATIENT_LIST_IvYGAP[0:int(NUM_PATIENT_ALL * 0.33)]


    TRAIN_LIST_IvYGAP = []
    TEST_LIST_IvYGAP = []
    for i in range(excel_wsi_IvYGAP.shape[0]):  # 2612
        if surv:
            if excel_wsi_IvYGAP[:, 0][i] in TRAIN_PATIENT_LIST_IvYGAP and (isinstance(excel_wsi_IvYGAP[i, 14], int) or isinstance(excel_wsi_IvYGAP[i, 14], float)) and not np.isnan(excel_wsi_IvYGAP[i, 14]):
                TRAIN_LIST_IvYGAP.append(excel_wsi_IvYGAP[i, :])
            elif excel_wsi_IvYGAP[:, 0][i] in TEST_PATIENT_LIST_IvYGAP and (isinstance(excel_wsi_IvYGAP[i, 14], int) or isinstance(excel_wsi_IvYGAP[i, 14], float)) and not np.isnan(excel_wsi_IvYGAP[i, 14]):
                TEST_LIST_IvYGAP.append(excel_wsi_IvYGAP[i, :])
        else:
            if excel_wsi_IvYGAP[:, 0][i] in TRAIN_PATIENT_LIST_IvYGAP:
                TRAIN_LIST_IvYGAP.append(excel_wsi_IvYGAP[i, :])
            elif excel_wsi_IvYGAP[:, 0][i] in TEST_PATIENT_LIST_IvYGAP:
                TEST_LIST_IvYGAP.append(excel_wsi_IvYGAP[i, :])
    LIST_IvYGAP = TRAIN_LIST_IvYGAP if phase == 'Train' else TEST_LIST_IvYGAP

    # #### tiantan
    excel_label_wsi_tiantan = pd.read_excel(opt['label_path'] + 'tiantan.xlsx', sheet_name='Sheet1', header=0)
    excel_wsi_tiantan = excel_label_wsi_tiantan.values
    PATIENT_LIST_tiantan = excel_wsi_tiantan[:, 0]
    PATIENT_LIST_tiantan = np.unique(PATIENT_LIST_tiantan)
    np.random.shuffle(PATIENT_LIST_tiantan)

    NUM_PATIENT_ALL = len(PATIENT_LIST_tiantan)
    if opt['fold'] == 0:
        TRAIN_PATIENT_LIST_tiantan = PATIENT_LIST_tiantan[0:int(NUM_PATIENT_ALL * 0.67)]
        TEST_PATIENT_LIST_tiantan = PATIENT_LIST_tiantan[int(NUM_PATIENT_ALL * 0.67):]
    elif opt['fold'] == 1:
        TRAIN_PATIENT_LIST_tiantan = np.concatenate(
            (PATIENT_LIST_tiantan[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST_tiantan[int(NUM_PATIENT_ALL * 0.67):]))
        TEST_PATIENT_LIST_tiantan = PATIENT_LIST_tiantan[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
    elif opt['fold'] == 2:
        TRAIN_PATIENT_LIST_tiantan = PATIENT_LIST_tiantan[int(NUM_PATIENT_ALL * 0.33):]
        TEST_PATIENT_LIST_tiantan = PATIENT_LIST_tiantan[0:int(NUM_PATIENT_ALL * 0.33)]

    TRAIN_LIST_tiantan = []
    TEST_LIST_tiantan = []
    for i in range(excel_wsi_tiantan.shape[0]):  # 2612
        if excel_wsi_tiantan[:, 0][i] in TRAIN_PATIENT_LIST_tiantan:
            TRAIN_LIST_tiantan.append(excel_wsi_tiantan[i, :])
        elif excel_wsi_tiantan[:, 0][i] in TEST_PATIENT_LIST_tiantan:
            TEST_LIST_tiantan.append(excel_wsi_tiantan[i, :])
    for i in range(len(TRAIN_LIST_tiantan)):
        a=TRAIN_LIST_tiantan[i][0]
        b=TRAIN_LIST_tiantan[i][1]

        if a <10:
            TRAIN_LIST_tiantan[i][0]='000'+str(a)
        elif a <100:
            TRAIN_LIST_tiantan[i][0] = '00' + str(a)
        elif a <1000:
            TRAIN_LIST_tiantan[i][0] = '0' + str(a)
        else:
            TRAIN_LIST_tiantan[i][0] = str(a)

        if b <10:
            TRAIN_LIST_tiantan[i][1]='000'+str(b)
        elif b <100:
            TRAIN_LIST_tiantan[i][1] = '00' + str(b)
        elif b <1000:
            TRAIN_LIST_tiantan[i][1] = '0' + str(b)
        else:
            TRAIN_LIST_tiantan[i][1] = str(b)
    for i in range(len(TEST_LIST_tiantan)):
        a=TEST_LIST_tiantan[i][0]
        b=TEST_LIST_tiantan[i][1]

        if a <10:
            TEST_LIST_tiantan[i][0]='000'+str(a)
        elif a <100:
            TEST_LIST_tiantan[i][0] = '00' + str(a)
        elif a <1000:
            TEST_LIST_tiantan[i][0] = '0' + str(a)
        else:
            TEST_LIST_tiantan[i][0] = str(a)

        if b <10:
            TEST_LIST_tiantan[i][1]='000'+str(b)
        elif b <100:
            TEST_LIST_tiantan[i][1] = '00' + str(b)
        elif b <1000:
            TEST_LIST_tiantan[i][1] = '0' + str(b)
        else:
            TEST_LIST_tiantan[i][1] = str(b)
    LIST_tiantan = TRAIN_LIST_tiantan if phase == 'Train' else TEST_LIST_tiantan

    #### CPTAC
    excel_label_wsi_CPTAC = pd.read_excel(opt['label_path'] + 'CPTAC.xlsx', sheet_name='Sheet1', header=0)
    excel_wsi_CPTAC = excel_label_wsi_CPTAC.values
    PATIENT_LIST_CPTAC = excel_wsi_CPTAC[:, 0]
    PATIENT_LIST_CPTAC = np.unique(PATIENT_LIST_CPTAC)
    np.random.shuffle(PATIENT_LIST_CPTAC)
    NUM_PATIENT_ALL = len(PATIENT_LIST_CPTAC)
    if opt['fold'] == 0:
        TRAIN_PATIENT_LIST_CPTAC = PATIENT_LIST_CPTAC[0:int(NUM_PATIENT_ALL * 0.67)]
        TEST_PATIENT_LIST_CPTAC = PATIENT_LIST_CPTAC[int(NUM_PATIENT_ALL * 0.67):]
    elif opt['fold'] == 1:
        TRAIN_PATIENT_LIST_CPTAC = np.concatenate(
            (PATIENT_LIST_CPTAC[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST_CPTAC[int(NUM_PATIENT_ALL * 0.67):]))
        TEST_PATIENT_LIST_CPTAC = PATIENT_LIST_CPTAC[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
    elif opt['fold'] == 2:
        TRAIN_PATIENT_LIST_CPTAC = PATIENT_LIST_CPTAC[int(NUM_PATIENT_ALL * 0.33):]
        TEST_PATIENT_LIST_CPTAC = PATIENT_LIST_CPTAC[0:int(NUM_PATIENT_ALL * 0.33)]

    TRAIN_LIST_CPTAC = []
    TEST_LIST_CPTAC = []
    for i in range(excel_wsi_CPTAC.shape[0]):  # 2612
        if surv:
            if excel_wsi_CPTAC[:, 0][i] in TRAIN_PATIENT_LIST_CPTAC and (isinstance(excel_wsi_CPTAC[i, 14], int) or isinstance(excel_wsi_CPTAC[i, 14], float)) and not np.isnan(excel_wsi_CPTAC[i, 14]):
                TRAIN_LIST_CPTAC.append(excel_wsi_CPTAC[i, :])
            elif excel_wsi_CPTAC[:, 0][i] in TEST_PATIENT_LIST_CPTAC and (isinstance(excel_wsi_CPTAC[i, 14], int) or isinstance(excel_wsi_CPTAC[i, 14], float)) and not np.isnan(excel_wsi_CPTAC[i, 14]):
                TEST_LIST_CPTAC.append(excel_wsi_CPTAC[i, :])
        else:

            if excel_wsi_CPTAC[:, 0][i] in TRAIN_PATIENT_LIST_CPTAC:
                TRAIN_LIST_CPTAC.append(excel_wsi_CPTAC[i, :])
            elif excel_wsi_CPTAC[:, 0][i] in TEST_PATIENT_LIST_CPTAC:
                TEST_LIST_CPTAC.append(excel_wsi_CPTAC[i, :])
    LIST_CPTAC = TRAIN_LIST_CPTAC if phase == 'Train' else TEST_LIST_CPTAC

    #### cambridge
    excel_label_wsi_cam = pd.read_excel(opt['label_path'] + 'cambridge.xlsx', sheet_name='Sheet1', header=0)
    excel_wsi_cam = excel_label_wsi_cam.values
    PATIENT_LIST_cam = excel_wsi_cam[:, 0]
    PATIENT_LIST_cam = np.unique(PATIENT_LIST_cam)
    np.random.shuffle(PATIENT_LIST_cam)
    NUM_PATIENT_ALL = len(PATIENT_LIST_cam)
    if opt['fold'] == 0:
        TRAIN_PATIENT_LIST_cam = PATIENT_LIST_cam[0:int(NUM_PATIENT_ALL * 0.67)]
        TEST_PATIENT_LIST_cam = PATIENT_LIST_cam[int(NUM_PATIENT_ALL * 0.67):]
    elif opt['fold'] == 1:
        TRAIN_PATIENT_LIST_cam = np.concatenate(
            (PATIENT_LIST_cam[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST_cam[int(NUM_PATIENT_ALL * 0.67):]))
        TEST_PATIENT_LIST_cam = PATIENT_LIST_cam[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
    elif opt['fold'] == 2:
        TRAIN_PATIENT_LIST_cam = PATIENT_LIST_cam[int(NUM_PATIENT_ALL * 0.33):]
        TEST_PATIENT_LIST_cam = PATIENT_LIST_cam[0:int(NUM_PATIENT_ALL * 0.33)]

    TRAIN_LIST_cam = []
    TEST_LIST_cam = []
    for i in range(excel_wsi_cam.shape[0]):  # 2612
        if surv:
            if excel_wsi_cam[:, 0][i] in TRAIN_PATIENT_LIST_cam and (
                    isinstance(excel_wsi_cam[i, 14], int) or isinstance(excel_wsi_cam[i, 14],
                                                                        float)) and not np.isnan(
                excel_wsi_cam[i, 14]):
                TRAIN_LIST_cam.append(excel_wsi_cam[i, :])
            elif excel_wsi_cam[:, 0][i] in TEST_PATIENT_LIST_cam and (
                    isinstance(excel_wsi_cam[i, 14], int) or isinstance(excel_wsi_cam[i, 14],
                                                                        float)) and not np.isnan(
                excel_wsi_cam[i, 14]):
                TEST_LIST_cam.append(excel_wsi_cam[i, :])
        else:

            if excel_wsi_cam[:, 0][i] in TRAIN_PATIENT_LIST_cam:
                TRAIN_LIST_cam.append(excel_wsi_cam[i, :])
            elif excel_wsi_cam[:, 0][i] in TEST_PATIENT_LIST_cam:
                TEST_LIST_cam.append(excel_wsi_cam[i, :])
    LIST_cam = TRAIN_LIST_cam if phase == 'Train' else TEST_LIST_cam

    #### ZS
    excel_label_wsi_ZS = pd.read_excel(opt['label_path'] + 'Zhongshan.xlsx', header=0)
    excel_wsi_ZS = excel_label_wsi_ZS.values
    PATIENT_LIST_ZS = excel_wsi_ZS[:, 0]
    PATIENT_LIST_ZS = np.unique(PATIENT_LIST_ZS)
    np.random.shuffle(PATIENT_LIST_ZS)
    NUM_PATIENT_ALL = len(PATIENT_LIST_ZS)
    if opt['fold'] == 0:
        TRAIN_PATIENT_LIST_ZS = PATIENT_LIST_ZS[0:int(NUM_PATIENT_ALL * 0.67)]
        TEST_PATIENT_LIST_ZS = PATIENT_LIST_ZS[int(NUM_PATIENT_ALL * 0.67):]
    elif opt['fold'] == 1:
        TRAIN_PATIENT_LIST_ZS = np.concatenate(
            (PATIENT_LIST_ZS[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST_ZS[int(NUM_PATIENT_ALL * 0.67):]))
        TEST_PATIENT_LIST_ZS = PATIENT_LIST_ZS[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
    elif opt['fold'] == 2:
        TRAIN_PATIENT_LIST_ZS = PATIENT_LIST_ZS[int(NUM_PATIENT_ALL * 0.33):]
        TEST_PATIENT_LIST_ZS = PATIENT_LIST_ZS[0:int(NUM_PATIENT_ALL * 0.33)]

    TRAIN_LIST_ZS = []
    TEST_LIST_ZS = []
    for i in range(excel_wsi_ZS.shape[0]):  # 2612
        if surv:
            if excel_wsi_ZS[:, 0][i] in TRAIN_PATIENT_LIST_ZS and (
                    isinstance(excel_wsi_ZS[i, 14], int) or isinstance(excel_wsi_ZS[i, 14],
                                                                       float)) and not np.isnan(
                excel_wsi_ZS[i, 14]):
                TRAIN_LIST_ZS.append(excel_wsi_ZS[i, :])
            elif excel_wsi_ZS[:, 0][i] in TEST_PATIENT_LIST_ZS and (
                    isinstance(excel_wsi_ZS[i, 14], int) or isinstance(excel_wsi_ZS[i, 14],
                                                                       float)) and not np.isnan(
                excel_wsi_ZS[i, 14]):
                TEST_LIST_ZS.append(excel_wsi_ZS[i, :])
        else:
            if excel_wsi_ZS[:, 0][i] in TRAIN_PATIENT_LIST_ZS:
                TRAIN_LIST_ZS.append(excel_wsi_ZS[i, :])
            elif excel_wsi_ZS[:, 0][i] in TEST_PATIENT_LIST_ZS:
                TEST_LIST_ZS.append(excel_wsi_ZS[i, :])
    LIST_ZS = TRAIN_LIST_ZS if phase == 'Train' else TEST_LIST_ZS


    if surv:
        LIST = np.asarray(LIST_CPTAC + LIST_IvYGAP + LIST_GBMatch+ LIST_cam+ LIST_ZS)
        LIST_CPTAC = np.asarray(TEST_LIST_CPTAC)
        LIST_IvYGAP = np.asarray(TEST_LIST_IvYGAP)
        LIST_GBMatch = np.asarray(TEST_LIST_GBMatch)
        LIST_tiantan = np.asarray(TEST_LIST_tiantan)
        LIST_cam = np.asarray(TEST_LIST_cam)
        LIST_ZS = np.asarray(TEST_LIST_ZS)
    else:
        LIST = np.asarray(LIST_CPTAC + LIST_IvYGAP + LIST_GBMatch + LIST_tiantan+ LIST_cam+ LIST_ZS)
        LIST_CPTAC=np.asarray(TRAIN_LIST_CPTAC+TEST_LIST_CPTAC)
        LIST_IvYGAP = np.asarray(TRAIN_LIST_IvYGAP+TEST_LIST_IvYGAP)
        LIST_GBMatch = np.asarray(TRAIN_LIST_GBMatch+TEST_LIST_GBMatch)
        LIST_tiantan = np.asarray(TRAIN_LIST_tiantan + TEST_LIST_tiantan)
        LIST_cam = np.asarray(TRAIN_LIST_cam + TEST_LIST_cam)
        LIST_ZS = np.asarray(TRAIN_LIST_ZS + TEST_LIST_ZS)

    if phase == 'Train':
        PATIENT_LIST=np.concatenate((TRAIN_PATIENT_LIST_GBMatch,TRAIN_PATIENT_LIST_IvYGAP,TRAIN_PATIENT_LIST_CPTAC,TRAIN_PATIENT_LIST_tiantan,TRAIN_PATIENT_LIST_cam,TRAIN_PATIENT_LIST_ZS))
    else:
        PATIENT_LIST = np.concatenate((TEST_PATIENT_LIST_GBMatch, TEST_PATIENT_LIST_IvYGAP, TEST_PATIENT_LIST_CPTAC, TEST_PATIENT_LIST_tiantan, TEST_PATIENT_LIST_cam, TEST_PATIENT_LIST_ZS))
    return LIST,PATIENT_LIST,LIST_CPTAC,LIST_IvYGAP,LIST_GBMatch,LIST_tiantan,LIST_cam,LIST_ZS





def get_model_stage1(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_His = Mine_His(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_His_Grade_2016(opt).cuda(gpuID[0])

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_His, init_type='xavier', init_gain=1)

    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_His = torch.nn.DataParallel(Mine_model_His, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)

    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_His = torch.optim.Adam(Mine_model_His.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)

    ###############  fp16 #######################
    if opt['fp16']:
        from apex import amp
        Mine_model_init, opt_init = amp.initialize(models=Mine_model_init, optimizers=opt_init, opt_level="O1")
        Mine_model_His, opt_His = amp.initialize(models=Mine_model_His, optimizers=opt_His, opt_level="O1")
        Mine_model_Cls, opt_Cls = amp.initialize(models=Mine_model_Cls, optimizers=opt_Cls, opt_level="O1")

    return Mine_model_init, Mine_model_His, Mine_model_Cls, opt_init,  opt_His, opt_Cls



def get_model_PredMarkers_ours(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_molecular = Mine_molecular_predall(opt).cuda(gpuID[0])
    Mine_model_Graph = Label_correlation_predall(opt).cuda(gpuID[0])


    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_molecular, init_type='xavier', init_gain=1)

    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_molecular = torch.nn.DataParallel(Mine_model_molecular, device_ids=gpuID)
    Mine_model_Graph = torch.nn.DataParallel(Mine_model_Graph, device_ids=gpuID)


    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_molecular = torch.optim.Adam(Mine_model_molecular.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Graph = torch.optim.Adam(Mine_model_Graph.parameters(), opt['Network']['lr'], weight_decay=0.00001)



    return Mine_model_init,Mine_model_molecular,Mine_model_Graph, opt_init,opt_molecular, opt_Graph,



def get_model_endtoend(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_body = Mine_endtoend_body(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_Diag_endtoend(opt).cuda(gpuID[0])

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_body, init_type='xavier', init_gain=1)
    init_weights(Mine_model_Cls, init_type='xavier', init_gain=1)

    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_body = torch.nn.DataParallel(Mine_model_body, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)


    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_body = torch.optim.Adam(Mine_model_body.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)


    return Mine_model_init,Mine_model_body,Mine_model_Cls, opt_init, opt_body, opt_Cls


def get_model_endtoend_marker(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_body = Mine_endtoend_body(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_Diag_endtoend_marker(opt).cuda(gpuID[0])

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_body, init_type='xavier', init_gain=1)
    init_weights(Mine_model_Cls, init_type='xavier', init_gain=1)

    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_body = torch.nn.DataParallel(Mine_model_body, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)


    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_body = torch.optim.Adam(Mine_model_body.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)


    return Mine_model_init,Mine_model_body,Mine_model_Cls, opt_init, opt_body, opt_Cls


def get_model_predallendtoend(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_body = Mine_endtoend_body(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_Diag_predallendtoend(opt).cuda(gpuID[0])

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_body, init_type='xavier', init_gain=1)
    init_weights(Mine_model_Cls, init_type='xavier', init_gain=1)

    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_body = torch.nn.DataParallel(Mine_model_body, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)


    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_body = torch.optim.Adam(Mine_model_body.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)


    return Mine_model_init,Mine_model_body,Mine_model_Cls, opt_init, opt_body, opt_Cls


def get_model_endtoendCNN(opt):
    gpuID = opt['gpus']
    Mine_CNN_cls = CNN_cls(opt).cuda(gpuID[0])
    init_weights(Mine_CNN_cls, init_type='xavier', init_gain=1)
    Mine_CNN_cls = torch.nn.DataParallel(Mine_CNN_cls, device_ids=gpuID)

    opt_CNN_cls = torch.optim.Adam(Mine_CNN_cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)

    return Mine_CNN_cls, opt_CNN_cls

def get_model_endtoendCNN_marker(opt):
    gpuID = opt['gpus']
    Mine_CNN_cls = CNN_cls_marker(opt).cuda(gpuID[0])
    init_weights(Mine_CNN_cls, init_type='xavier', init_gain=1)
    Mine_CNN_cls = torch.nn.DataParallel(Mine_CNN_cls, device_ids=gpuID)

    opt_CNN_cls = torch.optim.Adam(Mine_CNN_cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)

    return Mine_CNN_cls, opt_CNN_cls


def get_model_predallCNN(opt):
    gpuID = opt['gpus']
    Mine_CNN_cls = CNN_cls_predall(opt).cuda(gpuID[0])
    init_weights(Mine_CNN_cls, init_type='xavier', init_gain=1)
    Mine_CNN_cls = torch.nn.DataParallel(Mine_CNN_cls, device_ids=gpuID)

    opt_CNN_cls = torch.optim.Adam(Mine_CNN_cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)

    return Mine_CNN_cls, opt_CNN_cls


def get_model(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
    Mine_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
    Mine_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
    Mine_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
    Mine_model_His = Mine_His(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_His_Grade(opt).cuda(gpuID[0])
    device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')
    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_IDH, init_type='xavier', init_gain=1)
    init_weights(Mine_model_1p19q, init_type='xavier', init_gain=1)
    init_weights(Mine_model_CDKN, init_type='xavier', init_gain=1)
    init_weights(Mine_model_His, init_type='xavier', init_gain=1)


    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_IDH = torch.nn.DataParallel(Mine_model_IDH, device_ids=gpuID)
    Mine_model_1p19q = torch.nn.DataParallel(Mine_model_1p19q, device_ids=gpuID)
    Mine_model_CDKN = torch.nn.DataParallel(Mine_model_CDKN, device_ids=gpuID)
    Mine_model_Graph = torch.nn.DataParallel(Mine_model_Graph, device_ids=gpuID)
    Mine_model_His = torch.nn.DataParallel(Mine_model_His, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)

    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_IDH = torch.optim.Adam(Mine_model_IDH.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_1p19q = torch.optim.Adam(Mine_model_1p19q.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_CDKN = torch.optim.Adam(Mine_model_CDKN.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Graph = torch.optim.Adam(Mine_model_Graph.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_His = torch.optim.Adam(Mine_model_His.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    # opt_Grade = torch.optim.Adam(Mine_model_Grade.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)

    ###############  fp16 #######################
    if opt['fp16']:
        from apex import amp
        Mine_model_init, opt_init = amp.initialize(models=Mine_model_init,optimizers=opt_init,opt_level="O1")
        Mine_model_IDH, opt_IDH = amp.initialize(models=Mine_model_IDH, optimizers=opt_IDH,opt_level="O1")
        Mine_model_1p19q, opt_1p19q = amp.initialize(models=Mine_model_1p19q, optimizers=opt_1p19q,opt_level="O1")
        Mine_model_CDKN, opt_CDKN = amp.initialize(models=Mine_model_CDKN, optimizers=opt_CDKN,opt_level="O1")
        Mine_model_Graph, opt_Graph = amp.initialize(models=Mine_model_Graph, optimizers=opt_Graph, opt_level="O1")
        Mine_model_His, opt_His = amp.initialize(models=Mine_model_His, optimizers=opt_His,opt_level="O1")
        # Mine_model_Grade, opt_Grade = amp.initialize(models=Mine_model_Grade, optimizers=opt_Grade,opt_level="O1")
        Mine_model_Cls, opt_Cls = amp.initialize(models=Mine_model_Cls, optimizers=opt_Cls, opt_level="O1")


    return  Mine_model_init,Mine_model_IDH,Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His\
        ,Mine_model_Cls,opt_init,opt_IDH,opt_1p19q,opt_CDKN,opt_Graph,opt_His,opt_Cls

def cal_correct_ENDTOEND(predicted,label,Name):
    FLAT_normal = False
    count = 0
    NA_cls = 4 if Name == 'DiagSim' else 6
    for i in range(label.detach().cpu().numpy().shape[0]):
        if label.detach().cpu().numpy()[i] != NA_cls:
            if count == 0:
                pred_final = predicted[i].unsqueeze(0)
                label_final = label[i].unsqueeze(0)
                count += 1
            else:
                pred_final = torch.cat((pred_final, predicted[i].unsqueeze(0)), 0)
                label_final = torch.cat((label_final, label[i].unsqueeze(0)), 0)
                count += 1
            FLAT_normal = True
        else:
            continue
    total=0
    if not FLAT_normal:
        correct=0
    else:
        correct=pred_final.eq(label_final.data).cpu().sum()
        total=pred_final.detach().cpu().numpy().shape[0]

    return correct,FLAT_normal,total

def cal_mole_correct(predicted,label):
    FLAT_normal = False
    count = 0
    for i in range(label.detach().cpu().numpy().shape[0]):
        if label.detach().cpu().numpy()[i] != 2:
            if count == 0:
                pred_final = predicted[i].unsqueeze(0)
                label_final = label[i].unsqueeze(0)
                count += 1
            else:
                pred_final = torch.cat((pred_final, predicted[i].unsqueeze(0)), 0)
                label_final = torch.cat((label_final, label[i].unsqueeze(0)), 0)
                count += 1
            FLAT_normal = True
        else:
            continue
    total=0
    if not FLAT_normal:
        correct=0
        sen=0
        spec=0
    else:
        correct=pred_final.eq(label_final.data).cpu().sum()
        total=pred_final.detach().cpu().numpy().shape[0]
        pred_final=pred_final.tolist()
        label_final = label_final.tolist()
        tn=0
        fp=0
        fn=0
        tp=0
        for k in range(len(pred_final)):
            if label_final[k] == 0 and pred_final[k] == 0:
                tn += 1
            if label_final[k] == 0 and pred_final[k] == 1:
                fp += 1
            if label_final[k] == 1 and pred_final[k] == 0:
                fn += 1
            if label_final[k] == 1 and pred_final[k] == 1:
                tp += 1
        sen = (tp) / (tp + fn + 0.000001)
        spec = (tn) / (tn + fp + 0.000001)


    return correct,FLAT_normal,total,sen,spec

def cal_Diag_correct(opt,label_Diag, pred_IDH, pred_1p19q, pred_CDKN, pred_His,pred_Grade):
    FLAT_normal = False
    count = 0
    for i in range(label_Diag.detach().cpu().numpy().shape[0]):
        if label_Diag.detach().cpu().numpy()[i] != 6:
            if count == 0:
                pred_IDH_final = pred_IDH[i].unsqueeze(0)
                pred_1p19q_final = pred_1p19q[i].unsqueeze(0)
                pred_CDKN_final = pred_CDKN[i].unsqueeze(0)
                pred_His_final = pred_His[i].unsqueeze(0)
                pred_Grade_final = pred_Grade[i].unsqueeze(0)
                label_Diag_final = label_Diag[i].unsqueeze(0)
                count += 1
            else:
                pred_IDH_final = torch.cat((pred_IDH_final, pred_IDH[i].unsqueeze(0)), 0)
                pred_1p19q_final = torch.cat((pred_1p19q_final, pred_1p19q[i].unsqueeze(0)), 0)
                pred_CDKN_final = torch.cat((pred_CDKN_final, pred_CDKN[i].unsqueeze(0)), 0)
                pred_His_final = torch.cat((pred_His_final, pred_His[i].unsqueeze(0)), 0)
                pred_Grade_final = torch.cat((pred_Grade_final, pred_Grade[i].unsqueeze(0)), 0)
                label_Diag_final = torch.cat((label_Diag_final, label_Diag[i].unsqueeze(0)), 0)
                count += 1
            FLAT_normal = True
        else:
            continue
    total = 0
    if not FLAT_normal:
        correct = 0
    else:
        _, pred_IDH_final = torch.max(pred_IDH_final.data, 1)
        _, pred_1p19q_final = torch.max(pred_1p19q_final.data, 1)
        _, pred_CDKN_final = torch.max(pred_CDKN_final.data, 1)
        _, pred_His_final = torch.max(pred_His_final.data, 1)
        _, pred_Grade_final = torch.max(pred_Grade_final.data, 1)
        pred_final=Diag_full(IDH=pred_IDH_final, p19q=pred_1p19q_final, CDKN=pred_CDKN_final, His=pred_His_final,Grade=pred_Grade_final)
        pred_final=pred_final.cuda(opt['gpus'][0])[0]
        correct = pred_final.eq(label_Diag_final.data).cpu().sum()
        total = count


    return correct, FLAT_normal, total

def saliency_predcls_gene(saliency_IDH_wt,saliency_IDH_mut,saliency_1p19q_codel,saliency_1p19q_noncodel,
                                                       saliency_CDKN_HOMDEL,saliency_CDKN_NonHOMDEL,saliency_G2,saliency_G3,saliency_G4,pred_Diag):
    a=1
    # if pred_Diag[0]==0:
    #     saliency_predcls=saliency_IDH_wt
    # elif pred_Diag[0]==1:
    #     saliency_predcls=saliency_IDH_mut*saliency_1p19q_noncodel*(saliency_CDKN_HOMDEL+saliency_G4)
    # elif pred_Diag[0]==2:
    #     saliency_predcls = saliency_IDH_mut * saliency_1p19q_noncodel * saliency_CDKN_NonHOMDEL *saliency_G3
    # elif pred_Diag[0]==3:
    #     saliency_predcls = saliency_IDH_mut * saliency_1p19q_noncodel * saliency_CDKN_NonHOMDEL *saliency_G2
    # elif pred_Diag[0]==4:
    #     saliency_predcls = saliency_IDH_mut * saliency_1p19q_codel * saliency_G3
    # elif pred_Diag[0]==5:
    #     saliency_predcls = saliency_IDH_mut * saliency_1p19q_codel * saliency_G2
    if pred_Diag[0]==0:
        saliency_predcls=saliency_IDH_wt
    elif pred_Diag[0]==1 or pred_Diag[0]==2 or pred_Diag[0]==3:
        saliency_predcls=saliency_IDH_mut*saliency_1p19q_noncodel
    elif pred_Diag[0]==4 or pred_Diag[0]==5:
        saliency_predcls = saliency_IDH_mut*saliency_1p19q_codel

    return saliency_predcls


def saliency_predclsSim_gene(saliency_IDH_wt,saliency_IDH_mut,saliency_1p19q_codel,saliency_1p19q_noncodel,
                                                       saliency_CDKN_HOMDEL,saliency_CDKN_NonHOMDEL,saliency_G2,saliency_G3,saliency_G4,pred_Diag):
    a=1
    if pred_Diag[0]==0:
        saliency_predcls=saliency_IDH_wt
    elif pred_Diag[0]==1:
        saliency_predcls=saliency_IDH_mut*saliency_1p19q_noncodel*(saliency_CDKN_HOMDEL+saliency_G4)
    elif pred_Diag[0]==2:
        saliency_predcls = saliency_IDH_mut * saliency_1p19q_noncodel * saliency_CDKN_NonHOMDEL *(saliency_G3+saliency_G2)
    elif pred_Diag[0]==3:
        saliency_predcls = saliency_IDH_mut * saliency_1p19q_codel



    return saliency_predcls




def cal_DiagSim_correct(opt,label_Diag, pred_IDH, pred_1p19q, pred_CDKN, pred_His):
    FLAT_normal = False
    count = 0
    for i in range(label_Diag.detach().cpu().numpy().shape[0]):
        if label_Diag.detach().cpu().numpy()[i] != 4:
            if count == 0:
                pred_IDH_final = pred_IDH[i].unsqueeze(0)
                pred_1p19q_final = pred_1p19q[i].unsqueeze(0)
                pred_CDKN_final = pred_CDKN[i].unsqueeze(0)
                pred_His_final = pred_His[i].unsqueeze(0)
                label_Diag_final = label_Diag[i].unsqueeze(0)
                count += 1
            else:
                pred_IDH_final = torch.cat((pred_IDH_final, pred_IDH[i].unsqueeze(0)), 0)
                pred_1p19q_final = torch.cat((pred_1p19q_final, pred_1p19q[i].unsqueeze(0)), 0)
                pred_CDKN_final = torch.cat((pred_CDKN_final, pred_CDKN[i].unsqueeze(0)), 0)
                pred_His_final = torch.cat((pred_His_final, pred_His[i].unsqueeze(0)), 0)
                label_Diag_final = torch.cat((label_Diag_final, label_Diag[i].unsqueeze(0)), 0)
                count += 1
            FLAT_normal = True
        else:
            continue
    total = 0
    if not FLAT_normal:
        correct = 0
    else:
        _, pred_IDH_final = torch.max(pred_IDH_final.data, 1)
        _, pred_1p19q_final = torch.max(pred_1p19q_final.data, 1)
        _, pred_CDKN_final = torch.max(pred_CDKN_final.data, 1)
        _, pred_His_final = torch.max(pred_His_final.data, 1)
        pred_final=Diag_Simple(IDH=pred_IDH_final, p19q=pred_1p19q_final, CDKN=pred_CDKN_final, His=pred_His_final)
        pred_final=pred_final.cuda(opt['gpus'][0])[0]
        correct = pred_final.eq(label_Diag_final.data).cpu().sum()
        total = count
    return correct, FLAT_normal, total

def Diag_Simple(IDH, p19q, CDKN, His):
    """
    tensor:[BS]
    """
    Diag=[]
    for i in range(IDH.detach().cpu().numpy().shape[0]):
        if IDH[i] == 0:
            Diag.append(0)   # G4 GBM
        elif IDH[i] == 1:
            if p19q[i] == 1:
                Diag.append(3) # O
            elif p19q[i] == 0:
                if CDKN[i] == 1 or His[i] == 2:
                    Diag.append(1)  # G4 A
                else:
                    Diag.append(2)  # G23 A
    return torch.from_numpy(np.array(Diag))


def imp_gene(opt,img):
    gpuID = opt['gpus']
    batchSize=img.detach().cpu().numpy().shape[0]
    Name_cluster = opt['name'].split('_')
    if 'clstoken' in Name_cluster:
        imp_his=torch.ones(batchSize,opt['fixdim']+1)
        imp_grade=torch.ones(batchSize,opt['fixdim']+1)
    else:
        imp_his = torch.ones(batchSize, opt['fixdim'] )
        imp_grade = torch.ones(batchSize, opt['fixdim'] )

    imp_his =imp_his.cuda(gpuID[0])
    imp_grade = imp_grade.cuda(gpuID[0])
    return imp_his, imp_grade
def WHO2021_str_to_int(arr):
    map={'Grade 4 GBM':0,'Grade 4 Astrocytoma':1,'Grade 3 Astrocytoma':2,'Grade 2 Astrocytoma':3,'Grade 3 Oligodendroglioma':4,'Grade 2 Oligodendroglioma':5}
    out=[]
    for i in range(arr.shape[0]):
        out.append(map[arr[i]])
    return out
def WHO2007_str_to_int(arr):
    map={'Grade II':0,'Grade III':1,'Grade IV':2}
    out=[]
    for i in range(arr.shape[0]):
        out.append(map[arr[i]])
    return out
def get_model_stage2(opt):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt).cuda(gpuID[0])
    Mine_model_molecular = Mine_molecular(opt).cuda(gpuID[0])
    Mine_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
    Mine_model_His = Mine_His(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_His_Grade_2016(opt).cuda(gpuID[0])

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_His, init_type='xavier', init_gain=1)
    init_weights(Mine_model_molecular, init_type='xavier', init_gain=1)

    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_molecular = torch.nn.DataParallel(Mine_model_molecular, device_ids=gpuID)
    Mine_model_Graph = torch.nn.DataParallel(Mine_model_Graph, device_ids=gpuID)
    Mine_model_His = torch.nn.DataParallel(Mine_model_His, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)

    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_molecular = torch.optim.Adam(Mine_model_molecular.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Graph = torch.optim.Adam(Mine_model_Graph.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_His = torch.optim.Adam(Mine_model_His.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)


    return Mine_model_init,Mine_model_molecular,Mine_model_Graph, Mine_model_His, Mine_model_Cls, opt_init,opt_molecular, opt_Graph,opt_His, opt_Cls
def saliency_map_read_stage2(opt,file_name):

    for i in range(len(file_name)):
        if i == 0:
            saliency_map_His = np.expand_dims(
                np.load('./saliency/dynamic/'+opt['hispretrain']+'/His/' + file_name[i] + '.npy', allow_pickle=True), 0)
            saliency_map_Grade = np.expand_dims(
                np.load('./saliency/dynamic/'+opt['hispretrain']+'/Grade/' + file_name[i] + '.npy', allow_pickle=True), 0)
        else:
            saliency_map_His = np.concatenate((saliency_map_His, np.expand_dims(
                np.load('./saliency/dynamic/'+opt['hispretrain']+'/His/' + file_name[i] + '.npy', allow_pickle=True), 0)), 0)
            saliency_map_Grade = np.concatenate((saliency_map_Grade, np.expand_dims(
                np.load('./saliency/dynamic/'+opt['hispretrain']+'/Grade/' + file_name[i] + '.npy', allow_pickle=True), 0)), 0)

    return saliency_map_His,saliency_map_Grade
def Diag_full(IDH, p19q, CDKN, His,Grade):
    """
    tensor:[BS]
    """
    Diag=[]
    for i in range(IDH.detach().cpu().numpy().shape[0]):
        if IDH[i] == 0:
            Diag.append(0)   # G4 GBM
        elif IDH[i] == 1:
            if p19q[i] == 1:
                if Grade[i]==0:
                    Diag.append(5)   # G2 Oligo
                else:
                    Diag.append(4)  # G3 Oligo
            elif p19q[i] == 0:
                if CDKN[i] == 1 or His[i] == 2:
                    Diag.append(1)  # G4 A
                else:
                    if Grade[i] == 0:
                        Diag.append(3)   # G2 A
                    else:
                        Diag.append(2) # G3 A

    return torch.from_numpy(np.array(Diag))
if __name__ == "__main__":

    loss=FocalLoss()
    # pred=torch.from_numpy(np.asarray([[-0.2,-0.5]])).float()
    # label=torch.from_numpy(np.asarray([0])).long()
    # my_loss=loss(pred,label)
