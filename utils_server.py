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
import numpy as np
import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from PIL import Image
import cv2

import cv2
import math
import h5py
from model import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = 250000000000
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

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)

def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

def generate_read_details_tiantan(root):
    WSIextract_mine_root = root + r'extract_224/'
    CLAM_coor_root = root + 'refine/patches/'
    wsi_info= root + 'wsi_info/'
    wsi_info_file=os.listdir(wsi_info)
    wsi_info_file.sort()
    if not os.path.exists(root+'read_details'):
        os.makedirs(root+'read_details')
    for i in range(len(wsi_info_file)):
        coords_all = h5py.File(CLAM_coor_root + wsi_info_file[i])['coords'][:]
        patch_extract_mine = os.listdir(WSIextract_mine_root+ wsi_info_file[i][:-3])
        MPP = h5py.File(wsi_info + wsi_info_file[i] )['MPP'][()]
        relative_MPP = MPP / 0.5

        PATCH_SIZE_revise = np.int64(512 / relative_MPP)
        Num_cluster = 0
        point_num = coords_all.shape[0]
        points_center_corrd = coords_all + 256

        FLAT_w = points_center_corrd[0, 0]
        FLAT_h = points_center_corrd[0, 1]
        claster_corrds = {'0': []}
        mine_cluster = {'0': []}
        for k in range(point_num):
            w_point = points_center_corrd[k, 0]
            h_point = points_center_corrd[k, 1]
            if w_point == FLAT_w:
                claster_corrds[str(Num_cluster)].append([w_point, h_point])
                continue
            if w_point > FLAT_w:
                FLAT_w = w_point
                claster_corrds[str(Num_cluster)].append([w_point, h_point])
                continue
            if w_point < FLAT_w:
                FLAT_w = w_point
                Num_cluster += 1
                claster_corrds[str(Num_cluster)] = []
                mine_cluster[str(Num_cluster)] = []
                claster_corrds[str(Num_cluster)].append([w_point, h_point])

        Num_patch = len(patch_extract_mine)

        for j in range(Num_patch):
            w_index = int(patch_extract_mine[j].split('.')[0].split('_')[0])
            h_index = int(patch_extract_mine[j].split('.')[0].split('_')[1])
            center_patch_w = w_index * PATCH_SIZE_revise + int(PATCH_SIZE_revise / 2)
            center_patch_h = h_index * PATCH_SIZE_revise + int(PATCH_SIZE_revise / 2)

            patch_cluster_affliate = None
            Num_patch_dis = 0

            for m in range(Num_cluster + 1):
                point_sub_cluster = claster_corrds[str(m)]
                for n in range(len(point_sub_cluster)):
                    point_w = point_sub_cluster[n][0]
                    point_h = point_sub_cluster[n][1]
                    Distance = math.sqrt((center_patch_w - point_w) ** 2 + (center_patch_h - point_h) ** 2)
                    if Distance <= 512:
                        Num_patch_dis += 1
                    if Num_patch_dis == 2:
                        patch_cluster_affliate = m
                        if w_index < 10:
                            w_index = '00' + str(w_index)
                        elif w_index < 100:
                            w_index = '0' + str(w_index)
                        else:
                            w_index = str(w_index)
                        if h_index < 10:
                            h_index = '00' + str(h_index)
                        elif h_index < 100:
                            h_index = '0' + str(h_index)
                        else:
                            h_index = str(h_index)
                        mine_cluster[str(patch_cluster_affliate)].append(w_index + '_' + h_index)
                        break
                else:
                    continue
                break

        temp_list = []
        for key, value in enumerate(mine_cluster):
            mine_cluster[value].sort()
            temp_list += mine_cluster[value]

        ################Part 2
        reading_list = np.array(temp_list)
        Num_cluster = 0
        point_num = reading_list.shape[0]
        points_center_corrd = reading_list
        FLAT_w = int(points_center_corrd[0].split('_')[0])
        FLAT_h = int(points_center_corrd[0].split('_')[1])
        claster_corrds = {'0': []}
        for k in range(point_num):
            w_point = int(points_center_corrd[k].split('_')[0])
            h_point = int(points_center_corrd[k].split('_')[1])
            if w_point == FLAT_w:
                claster_corrds[str(Num_cluster)].append([w_point, h_point])
                continue
            # FLAT_w=w_point
            if w_point > FLAT_w and (w_point - FLAT_w) <= 6:
                FLAT_w = w_point
                claster_corrds[str(Num_cluster)].append([w_point, h_point])
                continue
            if w_point < FLAT_w or (w_point - FLAT_w) > 6:
                FLAT_w = w_point
                Num_cluster += 1
                claster_corrds[str(Num_cluster)] = []
                claster_corrds[str(Num_cluster)].append([w_point, h_point])
        a=1
        del_value = []
        for key, value in enumerate(claster_corrds):
            claster_corrds[value] = np.asarray(claster_corrds[value])
        for nn in range(len(del_value)):
            del claster_corrds[del_value[nn]]
        patch_length = []
        value_name = []
        for key, value in enumerate(claster_corrds):
            patch_length.append(claster_corrds[value].shape[0])
            value_name.append(value)
        save_dict = []
        sort_0 = np.argsort(np.asarray(patch_length))
        if len(value_name):
            for nn in range(1):
                save_dict.append(claster_corrds[value_name[sort_0[len(value_name) - nn - 1]]])
        np.save(root+'/read_details/' + wsi_info_file[i][:-3] + '.npy', save_dict)
        a = 1

import net
from dataset_mine import dataset_preprocess
def feature_generation(root):

    Res_pretrain = net.Res50_pretrain().cpu()
    Res_pretrain.eval()

    set = dataset_preprocess(root=root)
    dataLoader = DataLoader(set, batch_size=1, shuffle=False)
    data_bar = tqdm(dataLoader, disable=True)
    a=set.__len__()
    for packs in data_bar:
        img = packs[0][0]  # (N,3,224,224)
        imgPath = packs[1][0]
        patches_coor = packs[2][0]  # list N,2

        img = img.cpu()

        features_WSI=[]
        for i in range(2500):
            feature=Res_pretrain(torch.unsqueeze(img[i],dim=0))
            feature=feature.detach().cpu().numpy()
            features_WSI.append(feature)
        features_WSI=np.asarray(features_WSI) #(N,1024)
        feature_save = np.expand_dims(features_WSI, axis=0)
        feature_save = np.float16(feature_save)

        patches_coor = patches_coor.detach().numpy()
        if not os.path.exists(root + 'Res50_feature_2500_fixdim0_norm_0/'):
            os.makedirs(root + 'Res50_feature_2500_fixdim0_norm_0/')
        with h5py.File(root + 'Res50_feature_2500_fixdim0_norm_0/' + imgPath + '.h5', 'w') as f:
            f['Res_feature'] = feature_save
            f['patches_coor'] = patches_coor

def get_model_stage2(opt):

    Mine_model_init = Mine_init(opt).cpu()
    Mine_model_molecular = Mine_molecular(opt).cpu()
    Mine_model_Graph = Label_correlation_Graph(opt).cpu()
    Mine_model_His = Mine_His(opt).cpu()
    Mine_model_Cls = Cls_His_Grade_2016(opt).cpu()

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_His, init_type='xavier', init_gain=1)
    init_weights(Mine_model_molecular, init_type='xavier', init_gain=1)




    return Mine_model_init,Mine_model_molecular,Mine_model_Graph, Mine_model_His, Mine_model_Cls


def saliency_map_read_stage2(root):


    saliency_map_His = np.expand_dims(
        np.load(root+'/saliency/His/' + os.listdir(root+'/Res50_feature_2500_fixdim0_norm_0')[0][0:-3] + '.npy', allow_pickle=True), 0)
    saliency_map_Grade = np.expand_dims(
        np.load(root+'/saliency/Grade/' + os.listdir(root+'/Res50_feature_2500_fixdim0_norm_0')[0][0:-3] + '.npy', allow_pickle=True), 0)


    return saliency_map_His,saliency_map_Grade



def generate_saliency(Mine_model_init,Mine_model_His,Mine_model_Cls,trainDataset,root):

    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()

    if not os.path.exists(root+'/saliency/Grade/'):
        os.makedirs(root+'/saliency/Grade/')

    if not os.path.exists(root+'/saliency/His/'):
        os.makedirs(root+'/saliency/His/')
    trainLoader = DataLoader(trainDataset, batch_size=1,shuffle=True)


    train_bar = tqdm(trainLoader,disable=True)
    count = 0
    for packs in train_bar:
        img = packs[0]
        count += 1
        img = img.cpu()
        saliency_map_His = torch.ones(1, 2500)
        saliency_map_Grade = torch.ones(1, 2500)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cpu()
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cpu()


        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature, saliency_map_His, saliency_map_Grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4,_ = Mine_model_Cls(
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
        np.save(root+'/saliency/Grade/' + os.listdir(root+'/Res50_feature_2500_fixdim0_norm_0')[0][0:-3] + '.npy', saliency_final_Grade)
        np.save(root+'/saliency/His/' + os.listdir(root+'/Res50_feature_2500_fixdim0_norm_0')[0][0:-3] + '.npy', saliency_final_His)

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

import cmaps
def norm_mine(weight,num_patch):
    if num_patch<=2500:
        N_biorepet = int(2500 / num_patch)
        weight_0=weight[0:num_patch]
        if N_biorepet>1:
            for j in range(N_biorepet-1):
                weight_0+=weight[(j+1)*num_patch:(j+2)*num_patch]
        weight_0=weight_0/N_biorepet

        # min_w=np.min(weight_0)
        # max_w = np.max(weight_0)
        # weight_0=(weight_0-min_w)/(max_w-min_w)
    else:

        weight_0 = np.zeros(shape=(num_patch), dtype=np.float64)
        ori_list=[]
        for i in range(2500):
            ori_list.append(int(np.around(i*(num_patch/2500))))
            weight_0[int(np.around(i * (num_patch / 2500)))] = weight[i]
        for i in range (num_patch):
            if i  not in ori_list:
                for m in range(-3,3):
                    if  weight_0[i+m] :
                        weight_0[i]=weight_0[i+m]
                        break

        # min_w = np.min(weight_0)
        # max_w = np.max(weight_0)
        # weight_0 = (weight_0 - min_w) / (max_w - min_w)

    return weight_0



def reconstruct_tiantan(root,svs_name):
    if not os.path.exists(root + '/reconstruct'):
        os.makedirs(root + '/reconstruct')
    wsi_w = h5py.File(root +'/wsi_info.h5')['wsi_w'][()]
    wsi_h = h5py.File(root +'/wsi_info.h5')['wsi_h'][()]
    MPP = h5py.File(root +'/wsi_info.h5')['MPP'][()]
    relative_MPP = MPP / 0.5
    PATCH_SIZE_revise = np.int64(512 / relative_MPP)

    wsi_w = np.int64(wsi_w*(224/PATCH_SIZE_revise))+1
    wsi_h = np.int64(wsi_h*(224/PATCH_SIZE_revise))+1

    wsi_reconstruct = np.ones(shape=(wsi_h, wsi_w, 3), dtype=np.uint8)*255
    imgs_path=os.listdir(root+'extract_224/')

    for i in range(len(imgs_path)):
        img_patch = plt.imread(root+'extract_224/'+imgs_path[i])
        im = Image.fromarray(img_patch)
        im = im.resize((224, 224))
        img_patch=np.asarray(im)
        width_index=np.int64(imgs_path[i].split('.')[0].split('_')[0])
        height_index = np.int64(imgs_path[i].split('.')[0].split('_')[1])
        wsi_reconstruct[height_index*224:(height_index+1)*224,width_index*224:(width_index+1)*224,:]=img_patch

    wsi_reconstruct = Image.fromarray(wsi_reconstruct)
    if wsi_h>=wsi_w:
        wsi_reconstruct = wsi_reconstruct.resize(( int( wsi_w/ wsi_h * 1024),1024))
    else:
        wsi_reconstruct = wsi_reconstruct.resize((1024, int(wsi_h / wsi_w * 1024)))
    wsi_reconstruct.save(root+'/reconstruct/'+svs_name+'.jpg')
    z=1

class Saver():
  def __init__(self, opt):
    self.logDir = opt['logDir']
    self.n_ep_save = opt['n_ep_save']
    self.writer = SummaryWriter(logdir=self.logDir)

  def write_scalars(self, ep, lossdict):
    # Todo Save images
    for loss_key, loss_value in lossdict.items():
        self.writer.add_scalar(loss_key, loss_value, ep)


  def write_maps(self, ep, map_dict):
    for name,map in map_dict.items():
      if len(map.shape)==2:
        map = map[np.newaxis,...]
      if map.shape[0] == 1:
        map = np.concatenate((map, map, map), axis=0)
      self.writer.add_image('map/'+name, map, ep)


  def write_log(self, ep, lossdict, Name):
    logpath = os.path.join(self.logDir, Name + '.log')
    title = 'epochs,'
    vals = '%d,'%(ep)
    for loss_key, loss_value in lossdict.items():
      title = title + loss_key + ','
      vals = vals + '%4f,'% (loss_value)
    title = title[:-1] + '\n'
    vals = vals[:-1] + '\n'
    if ep==self.n_ep_save-1:
      saveFile = open(logpath, "w")
      saveFile.write(title)
      saveFile.write(vals)
    else:
      saveFile = open(logpath, "a")
      saveFile.write(vals)
    saveFile.close()




  def write_imagegroup(self, ep, images, basename, key):
    # images: tensor Bx3xHxW or Bx1xHxW or BxHxW
    if len(images.shape) == 3:
      images = torch.unsqueeze(images, 1)
      images = torch.cat([images, images, images], 1)
    elif images.shape[1] == 1:
      images = torch.cat([images, images, images], 1)
    image_dis = torchvision.utils.make_grid(images, nrow=7)
    self.writer.add_image('map/' + key, image_dis, ep)
    image_dis2 = image_dis

    ndarr = image_dis2.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    savename = os.path.join(self.logDir, key + '_' + basename + '_' +str(ep) + '.png')
    if key == 'SAmap':
      ndarr = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
    cv2.imwrite(savename, ndarr)

    return ndarr



def diagnosis_2021(IDH,pq, CDKN,grade):
    if IDH=='WT':
        subtype_2021='glioblastoma'
    elif IDH=='Mutant':
        if pq=='codel':
            subtype_2021 = 'oligodendroglioma'
        elif pq =='non-codel':
            subtype_2021 ='astrocytoma'
        else:
            subtype_2021 = 'None'
    else:
        subtype_2021 = 'None'

    if IDH == 'WT':
        grade_2021='G4'
    elif IDH == 'Mutant':
        if pq == 'codel':
            if grade=='G2':
                grade_2021 = 'G2'
            else:
                grade_2021 = 'G3'
        elif pq == 'non-codel':
            if grade=='G4' or (CDKN==-1 or CDKN==-2):
                grade_2021 = 'G4'
            else:
                if grade == 'G2':
                    grade_2021 = 'G2'
                else:
                    grade_2021 = 'G3'
        else:
            grade_2021 = 'None'

    else:
        grade_2021 = 'None'
    return subtype_2021,grade_2021


def count_noisy_clean(LIST):
    # OA is originally noisy

    LIST_ori=LIST
    clean_patient_subtype= []
    clean_wsi_subtype = []
    noisy_patient_subtype = []
    noisy_wsi_subtype = []
    clean_patient_grade = []
    clean_wsi_grade = []
    noisy_patient_grade = []
    noisy_wsi_grade = []
    ### subtype
    for i in range(LIST_ori.shape[0]):
        if not LIST_ori[i,2]=='oligoastrocytoma':
            subtype_2007=LIST_ori[i,2]
            subtype_2021,_=diagnosis_2021(LIST_ori[i, 4], LIST_ori[i, 5], LIST_ori[i, 6],LIST_ori[i, 3])
            if subtype_2007==subtype_2021:
                clean_wsi_subtype.append(LIST_ori[i,:])
            else:
                noisy_wsi_subtype.append(LIST_ori[i,:])
            a=1
        print(i)
    ### grade
    for i in range(LIST_ori.shape[0]):
        grade_2007 = LIST_ori[i, 3]
        _,grade_2021=diagnosis_2021(LIST_ori[i, 4], LIST_ori[i, 5], LIST_ori[i, 6],LIST_ori[i, 3])
        if grade_2007==grade_2021:
            clean_wsi_grade.append(LIST_ori[i,:])
        else:
            noisy_wsi_grade.append(LIST_ori[i,:])
        a=1

    clean_wsi_subtype = np.array(clean_wsi_subtype)
    noisy_wsi_subtype = np.array(noisy_wsi_subtype)
    clean_wsi_grade = np.array(clean_wsi_grade)
    noisy_wsi_grade = np.array(noisy_wsi_grade)
    return clean_wsi_subtype, noisy_wsi_subtype,clean_wsi_grade, noisy_wsi_grade

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



def test_stage2(opt,Mine_model_init, Mine_model_His, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID,external=False, name='None'):

    file_name_array=[]
    gt_His_array=[]
    pred_His_ori_array=[]
    pred_His_array=[]
    gt_Grade_array=[]
    pred_Grade_ori_array=[]
    pred_Grade_array=[]

    gt_IDH_array=[]
    pred_IDH_ori_array=[]
    pred_IDH_array=[]

    gt_1p19q_array=[]
    pred_1p19q_ori_array=[]
    pred_1p19q_array=[]

    gt_CDKN_array=[]
    pred_CDKN_ori_array=[]
    pred_CDKN_array=[]

    gt_Diag_array=[]
    pred_Diag_ori_array=[]
    pred_Diag_array=[]

    gt_DiagSim_array=[]
    pred_DiagSim_ori_array=[]
    pred_DiagSim_array=[]
    test_bar = tqdm(dataloader)
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name = packs[2]
        file_name_array.append(file_name[0])
        patient_name =  packs[3][0]
        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]
        label_IDH = label[:, 2]
        label_1p19q = label[:, 3]
        label_CDKN = label[:, 4]
        label_Diag_simple = label[:, 5]
        label_Diag = label[:, 6]

        saliency_map_His, saliency_map_Grade = saliency_map_read_stage2(opt, file_name)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,saliency_map_His, saliency_map_Grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
            encoded_His, encoded_Grade)

        ### WHO2007 prediction
        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His0 = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His0.tolist()  # [BS] A  O GBM //0 1 2
        gt_His = label_his.tolist()
        _, pred_Grade0 = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade0.tolist()  # [BS] A  O GBM //0 1 2
        gt_Grade = label_grade.tolist()
        pred_His_ori = F.softmax(pred_His_ori)
        pred_Grade_ori = F.softmax(pred_Grade_ori)

        ### molecular prediction
        encoded_IDH, encoded_1p19q, encoded_CDKN = Mine_model_molecular(init_feature)
        results_dict, saliency_IDH_wt, saliency_1p19q_codel, encoded_IDH0, encoded_1p19q0, encoded_CDKN0 = Mine_model_Graph(
            encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH_ori = results_dict['logits_IDH']
        pred_1p19q_ori = results_dict['logits_1p19q']
        pred_CDKN_ori = results_dict['logits_CDKN']
        _, pred_IDH0 = torch.max(pred_IDH_ori.data, 1)
        pred_IDH = pred_IDH0.tolist()
        gt_IDH = label_IDH.tolist()
        _, pred_1p19q0 = torch.max(pred_1p19q_ori.data, 1)
        pred_1p19q = pred_1p19q0.tolist()
        gt_1p19q = label_1p19q.tolist()
        _, pred_CDKN0 = torch.max(pred_CDKN_ori.data, 1)
        pred_CDKN = pred_CDKN0.tolist()
        gt_CDKN = label_CDKN.tolist()
        pred_IDH_ori = F.softmax(pred_IDH_ori)
        pred_1p19q_ori = F.softmax(pred_1p19q_ori)
        pred_CDKN_ori = F.softmax(pred_CDKN_ori)

        ### Diag prediction
        gt_Diag = label_Diag.tolist()
        pred_Diag = Diag_full(pred_IDH0, pred_1p19q0, pred_CDKN0, pred_His0, pred_Grade0).tolist()
        pred_Diag_ori = gene_Diag_ori(pred_IDH_ori[0], pred_1p19q_ori[0], pred_CDKN_ori[0], pred_His_ori[0],
                                      pred_Grade_ori[0])

        ### Diag Simple prediction
        gt_DiagSim = label_Diag_simple.tolist()
        pred_DiagSim = Diag_Simple(pred_IDH0, pred_1p19q0, pred_CDKN0, pred_His0).tolist()
        pred_DiagSim_ori = gene_DiagSim_ori(pred_IDH_ori[0], pred_1p19q_ori[0], pred_CDKN_ori[0], pred_His_ori[0])

        ############################ WSI calculate tntp
        gt_His_array.append(gt_His[0])
        pred_His_ori_array.append(pred_His_ori.detach().cpu().numpy()[0])
        pred_His_array.append(pred_His[0])
        gt_Grade_array.append(gt_Grade[0])
        pred_Grade_ori_array.append(pred_Grade_ori.detach().cpu().numpy()[0])
        pred_Grade_array.append(pred_Grade[0])
        gt_IDH_array.append(gt_IDH[0])
        pred_IDH_ori_array.append(pred_IDH_ori.detach().cpu().numpy()[0][1])
        pred_IDH_array.append(pred_IDH[0])
        gt_1p19q_array.append(gt_1p19q[0])
        pred_1p19q_ori_array.append(pred_1p19q_ori.detach().cpu().numpy()[0][1])
        pred_1p19q_array.append(pred_1p19q[0])
        gt_CDKN_array.append(gt_CDKN[0])
        pred_CDKN_ori_array.append(pred_CDKN_ori.detach().cpu().numpy()[0][1])
        pred_CDKN_array.append(pred_CDKN[0])
        gt_Diag_array.append(gt_Diag[0])
        pred_Diag_ori_array.append(pred_Diag_ori)
        pred_Diag_array.append(pred_Diag[0])
        gt_DiagSim_array.append(gt_DiagSim[0])
        pred_DiagSim_ori_array.append(pred_DiagSim_ori)
        pred_DiagSim_array.append(pred_DiagSim[0])


    if not external:
        h5name='./pred_files/ours/TCGA_fold'+str(opt['fold'])+'-WSI.h5'
        npyname = './pred_files/WSI_name/TCGA_fold'+str(opt['fold'])+'-WSI.npy'
    else:
        h5name = './pred_files/ours/' + name + '_fold' + str(opt['fold']) + '-WSI.h5'
        npyname = './pred_files/WSI_name/' + name + '_fold' + str(opt['fold']) + '-WSI.npy'

    np.save(npyname,np.array(file_name_array))
    with h5py.File(h5name, 'w') as f:
        f['gt_His_array'] = np.array(gt_His_array)
        f['pred_His_ori_array'] = np.array(pred_His_ori_array)
        f['pred_His_array'] = np.array(pred_His_array)
        f['gt_Grade_array'] = np.array(gt_Grade_array)
        f['pred_Grade_ori_array'] = np.array(pred_Grade_ori_array)
        f['pred_Grade_array'] = np.array(pred_Grade_array)

        f['gt_IDH_array'] = np.array(gt_IDH_array)
        f['pred_IDH_ori_array'] = np.array(pred_IDH_ori_array)
        f['pred_IDH_array'] =np.array( pred_IDH_array)

        f['gt_1p19q_array'] = np.array(gt_1p19q_array)
        f['pred_1p19q_ori_array'] =np.array( pred_1p19q_ori_array)
        f['pred_1p19q_array'] = np.array(pred_1p19q_array)

        f['gt_CDKN_array'] =np.array( gt_CDKN_array)
        f['pred_CDKN_ori_array'] = np.array(pred_CDKN_ori_array)
        f['pred_CDKN_array'] = np.array(pred_CDKN_array)

        f['gt_Diag_array'] = np.array(gt_Diag_array)
        f['pred_Diag_ori_array'] = np.array(pred_Diag_ori_array)
        f['pred_Diag_array'] = np.array(pred_Diag_array)

        f['gt_DiagSim_array'] = np.array(gt_DiagSim_array)
        f['pred_DiagSim_ori_array'] = np.array(pred_DiagSim_ori_array)
        f['pred_DiagSim_array'] = np.array(pred_DiagSim_array)


def test_stage2_stem(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,Mine_model_molecular, Mine_model_Graph,dataloader, gpuID):

    test_stage2(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,Mine_model_molecular,Mine_model_Graph, dataloader, gpuID)
