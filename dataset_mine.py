from __future__ import print_function
from skimage import io
import h5py
from utils import *



class Our_Dataset_stage2_3fold(Dataset):
    def __init__(self, phase,opt,external=None):
        super(Our_Dataset_stage2_3fold, self).__init__()
        self.opt = opt
        self.patc_bs=64
        self.phase=phase
        self.external = external
        excel_label_wsi = pd.read_excel(opt['label_path'] + 'TCGA.xlsx', sheet_name='wsi_level', header=0)
        excel_wsi = excel_label_wsi.values
        PATIENT_LIST = excel_wsi[:, 0]
        np.random.seed(self.opt['seed'])
        random.seed(self.opt['seed'])
        PATIENT_LIST = list(PATIENT_LIST)

        PATIENT_LIST = np.unique(PATIENT_LIST)
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL = len(PATIENT_LIST)
        if self.opt['fold'] == 0:
            TRAIN_PATIENT_LIST_TCGA = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.67)]
            TEST_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]
        elif self.opt['fold'] == 1:
            TRAIN_PATIENT_LIST_TCGA = np.concatenate(
                (PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]))
            TEST_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
        elif self.opt['fold'] == 2:
            TRAIN_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.33):]
            TEST_PATIENT_LIST_TCGA = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.33)]


        ###----end----other datasets

        self.TRAIN_LIST = []
        self.TEST_LIST = []
        for i in range(excel_wsi.shape[0]):  # 2612
            if excel_wsi[:, 0][i] in TRAIN_PATIENT_LIST_TCGA:
                self.TRAIN_LIST.append(excel_wsi[i, :])
            if excel_wsi[:, 0][i] in TEST_PATIENT_LIST_TCGA:
                self.TEST_LIST.append(excel_wsi[i, :])
        self.LIST = np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else np.asarray(self.TEST_LIST)




        self.train_iter_count = 0
        self.Flat = 0
        self.WSI_all = []



    def __getitem__(self, index):
        feature_all = self.read_feature(index)

        label = self.label_gene(index)

        return torch.from_numpy(np.array(feature_all)).float(), torch.from_numpy(label), self.LIST[index, 1], self.LIST[
            index, 0]


    def read_feature(self, index):
        if self.LIST[index, 0][0] == 'T':
            root = self.opt['dataDir'] + 'TCGA/Res50_feature_' + str(self.opt['fixdim']) + '_fixdim0_norm/'


        patch_all = h5py.File(root + self.LIST[index, 1] + '.h5')['Res_feature'][:]  # (1,N,1024)
        return patch_all[0]

    def label_gene(self, index):

        ###  WHO 2007 subtype of A/O/GBM
        if self.LIST[index, 2] == 'oligoastrocytoma':
            label_subtype = 3
        elif self.LIST[index, 2] == 'astrocytoma':
            label_subtype = 0
        elif self.LIST[index, 2] == 'oligodendroglioma':
            label_subtype = 1
        elif self.LIST[index, 2] == 'glioblastoma':
            label_subtype = 2

        ###  WHO 2007 Grading
        if self.LIST[index, 3] == 'G2':
            label_Grade = 0
        elif self.LIST[index, 3] == 'G3':
            label_Grade = 1
        elif self.LIST[index, 3] == 'G4':
            label_Grade = 2

        ###  molecular
        a = str(self.LIST[index, 5])
        if self.LIST[index, 4] == 'WT':
            label_IDH = 0
        elif self.LIST[index, 4] == 'Mutant':
            label_IDH = 1
        else:
            label_IDH = 2
        if self.LIST[index, 5] == 'non-codel':
            label_1p19q = 0
        elif self.LIST[index, 5] == 'codel':
            label_1p19q = 1
        else:
            label_1p19q = 2
        if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1:
            label_CDKN = 1
        elif self.LIST[index, 6] == 1 or self.LIST[index, 6] == 0:
            label_CDKN = 0
        else:
            label_CDKN = 2

        ###  Diag_simple: GBM,  G4 A , G2/3 A, G2/3 O, NA--> 0,1,2,3,4
        if label_IDH == 0:
            label_Diag_simple = 0  # G4 GBM
        elif label_IDH == 1:
            if label_1p19q == 1:
                label_Diag_simple = 3  # G2/3 Oligo
            elif label_1p19q == 0:
                if label_CDKN == 1 or label_Grade == 2:
                    label_Diag_simple = 1  # G4 A
                elif label_CDKN == 0:
                    label_Diag_simple = 2  # G2/3 A
                elif label_CDKN == 2:
                    label_Diag_simple = 4
            elif label_1p19q == 2:
                label_Diag_simple = 4
        elif label_IDH == 2:
            label_Diag_simple = 4

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

        label = np.asarray(
            [label_subtype, label_Grade, label_IDH, label_1p19q, label_CDKN, label_Diag_simple, label_Diag])

        return label

    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)

    def __len__(self):
        return self.LIST.shape[0]


class dataset_preprocess(Dataset):
    def __init__(self,root):
        super(dataset_preprocess, self).__init__()
        self.root=root
        ori_list=os.listdir(root+'read_details/')
        ori_list_temp=ori_list.copy()
        for i in range(len(ori_list)):
            if not np.load(root+'read_details/'+ori_list[i], allow_pickle=True).shape[0]:
                ori_list_temp.remove(ori_list[i])

        self.LIST= np.array(ori_list_temp)


    def __getitem__(self, index):

        patch_all,coor_all=self.read_img(index)

        return torch.from_numpy(np.array(patch_all)).float(), self.LIST[index][:-4],np.array(coor_all)

    def read_img(self,index):
        patch_all = []
        patch_all_ori=[]
        coor_all=[]
        coor_all_ori = []

        read_details=np.load(self.root+'read_details/'+self.LIST[index][:-4]+'.npy',allow_pickle=True)[0]
        num_patches = read_details.shape[0]

        max_num=2500
        Use_patch_num = num_patches if num_patches <= max_num else max_num
        if num_patches <= max_num:
            times=int(np.floor(max_num/num_patches))
            remaining=max_num % num_patches
            for i in range(Use_patch_num):
                img_temp=io.imread(self.root+'extract_224/'+self.LIST[index][:-4]+'/'+ str(read_details[i][0]) + '_' + str(read_details[i][1]) + '.jpg')
                img_temp = cv2.resize(img_temp, (224, 224))
                patch_all_ori.append(img_temp)
                coor_all_ori.append(read_details[i])
            patch_all=patch_all_ori
            coor_all = coor_all_ori

            ####### fixdim0
            if times>1:
                for k in range(times-1):
                    patch_all=patch_all+patch_all_ori
                    coor_all=coor_all+coor_all_ori
            if not remaining==0:
                patch_all = patch_all + patch_all_ori[0:remaining]
                coor_all = coor_all + coor_all_ori[0:remaining]

        else:
            for i in range(Use_patch_num):
                img_temp = io.imread(self.root+'extract_224/' + str(read_details[int(np.around(i*(num_patches/max_num)))][0])+'_'+str(read_details[int(np.around(i*(num_patches/max_num)))][1])+'.jpg')
                img_temp = cv2.resize(img_temp, (224, 224))
                patch_all.append(img_temp)
                coor_all.append(read_details[int(np.around(i*(num_patches/max_num)))])

        patch_all = np.asarray(patch_all)

        # data augmentation

        patch_all = patch_all.reshape(-1, 224, 3)  # (num_patches*224,224,3)
        patch_all = patch_all.reshape(-1, 224, 224, 3)  # (num_patches,224,224,3)
        patch_all = patch_all / 255.0

        ######normalization
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        patch_all[...,0]=(patch_all[...,0]-mean[0])/std[0]
        patch_all[..., 1] = (patch_all[..., 1] - mean[1]) / std[1]
        patch_all[..., 2] = (patch_all[..., 2] - mean[2]) / std[2]

        patch_all = np.transpose(patch_all, (0, 3, 1, 2))
        patch_all = patch_all.astype(np.float32)


        return patch_all,coor_all


    def __len__(self):
        return self.LIST.shape[0]

class Our_Dataset_stage1_3fold(Dataset):
    def __init__(self, phase,opt,external=None):
        super(Our_Dataset_stage1_3fold, self).__init__()
        self.opt = opt
        self.patc_bs=64
        self.phase=phase
        self.external=external

        excel_label_wsi = pd.read_excel(opt['label_path'] + 'TCGA.xlsx', sheet_name='wsi_level', header=0)
        excel_wsi = excel_label_wsi.values
        PATIENT_LIST = excel_wsi[:, 0]
        np.random.seed(self.opt['seed'])
        random.seed(self.opt['seed'])
        PATIENT_LIST = list(PATIENT_LIST)

        PATIENT_LIST = np.unique(PATIENT_LIST)  # 2296
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL = len(PATIENT_LIST)  # 952
        if self.opt['fold'] == 0:
            TRAIN_PATIENT_LIST_TCGA = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.67)]
            TEST_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]
        elif self.opt['fold'] == 1:
            TRAIN_PATIENT_LIST_TCGA = np.concatenate(
                (PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]))
            TEST_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.33):int(NUM_PATIENT_ALL * 0.67)]
        elif self.opt['fold'] == 2:
            TRAIN_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.33):]
            TEST_PATIENT_LIST_TCGA = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.33)]


        ###----end----other datasets

        self.TRAIN_LIST = []
        self.TEST_LIST = []
        for i in range(excel_wsi.shape[0]):  # 2612
            if excel_wsi[:, 0][i] in TRAIN_PATIENT_LIST_TCGA:
                self.TRAIN_LIST.append(excel_wsi[i, :])
            elif excel_wsi[:, 0][i] in TEST_PATIENT_LIST_TCGA:
                self.TEST_LIST.append(excel_wsi[i, :])
        self.LIST = np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else np.asarray(self.TEST_LIST)




        self.train_iter_count = 0
        self.Flat = 0
        self.WSI_all = []

    def __getitem__(self, index):
        feature_all = self.read_feature(index)

        label=self.label_gene(index)

        return torch.from_numpy(np.array(feature_all)).float(),torch.from_numpy(label),self.LIST[index, 1], self.LIST[index, 0]

    def read_feature(self, index):
        self.opt['fixdim']=2500
        if self.LIST[index, 0][0]=='T':
            root = self.opt['dataDir'] + 'TCGA/Res50_feature_' + str(self.opt['fixdim']) + '_fixdim0_norm/'

        patch_all = h5py.File(root + self.LIST[index, 1] + '.h5')['Res_feature'][:]  # (1,N,1024)
        return patch_all[0]


    def label_gene(self,index):


        ###  subtype of A/O/GBM
        if self.LIST[index, 2]=='oligoastrocytoma':
            label_subtype = 3
        elif self.LIST[index, 2] == 'astrocytoma':
            label_subtype = 0
        elif self.LIST[index, 2] == 'oligodendroglioma':
            label_subtype = 1
        elif self.LIST[index, 2] == 'glioblastoma':
            label_subtype = 2

        ###  Grading
        if self.LIST[index, 3]=='G2':
            label_Grade=0
        elif self.LIST[index, 3] == 'G3':
            label_Grade = 1
        elif self.LIST[index, 3] == 'G4':
            label_Grade=2

        if self.LIST[index, 4]=='WT':
            label_Diag = 0
        elif self.LIST[index, 5] == 'codel':
            label_Diag = 2
        else:
            label_Diag = 1

        label=np.asarray([label_subtype,label_Grade])

        return  label


    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)



    def __len__(self):
        return self.LIST.shape[0]
