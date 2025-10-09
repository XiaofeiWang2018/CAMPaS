import os.path

import argparse, time, random

import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import get_model_endtoend,get_model_endtoendCNN,saliency_predcls_gene,saliency_predclsSim_gene
from utils_server import *
from yaml.loader import SafeLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = 250000000000
from torch.utils.data import Dataset,DataLoader
from patch_WSI import main_fp
import platform
from dataset_mine import *
sysstr = platform.system()
if not (sysstr == "Linux"):
    OPENSLIDE_PATH = r'C:\ProgramData\Anaconda3\envs\torch1.10\Library\openslide-win64-20220806\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import OpenSlide, OpenSlideUnsupportedFormatError
else:
    from openslide import OpenSlide
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter
from matplotlib import cm

parser = argparse.ArgumentParser()
### Not input, just keept as default:
parser.add_argument('--source', type = str, default='DATA',help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=512,help='step_size')
parser.add_argument('--patch_size', type = int, default=512,help='patch_size')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--stitch', default=True, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,default='RESULTS', help='directory to save processed data')
parser.add_argument('--preset', default='tcga.csv', type=str,help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,help='downsample level at which to patch')
parser.add_argument('--dataset_name', type='dataset1', default=0,help='name of your under-processed dataset')
args = parser.parse_args()



root=r'my_data/'+args.dataset_name+'/'
files=os.listdir(root)
if not os.path.exists('./processed_data/'+args.dataset_name+'/wsi_info/'):
    os.makedirs('./processed_data/'+args.dataset_name+'/wsi_info/')
else:
    remove_all_file('./processed_data/'+args.dataset_name+'/wsi_info/')
for file in range(len(files)):
    print('Processing ',file,'-th WSI out of ',len(files))
    ###### 1. patching
    wsi_obj = OpenSlide(root+files[file])
    MPP =0.5
    PATCH_SIZE=512
    relative_MPP=MPP/0.5
    PATCH_SIZE_revise=np.int64(PATCH_SIZE/relative_MPP)
    width_whole, height_whole = wsi_obj.level_dimensions[0]
    prefix_dir_224 = './processed_data/'+args.dataset_name+'/extract_224/'+str(files[file][:-4]) +'/'

    with h5py.File('./processed_data/'+args.dataset_name+'/wsi_info/'+files[file][:-4]+'.h5', 'w') as f:
        f['wsi_w'] = width_whole
        f['wsi_h'] = height_whole
        f['MPP'] = MPP

    if not os.path.exists(prefix_dir_224):
        os.makedirs(prefix_dir_224)
    else:
        remove_all_file(prefix_dir_224)
    for i in range(np.int64(width_whole/PATCH_SIZE_revise)):
        for j in range(np.int64(height_whole / PATCH_SIZE_revise)):
            rgba_image_pil = wsi_obj.read_region((i*PATCH_SIZE_revise, j*PATCH_SIZE_revise), 0, (PATCH_SIZE_revise, PATCH_SIZE_revise))
            rgba_image = np.asarray(rgba_image_pil)[..., 0:3]
            r_ = rgba_image[..., 0]
            g_ = rgba_image[..., 1]
            b_ = rgba_image[..., 2]
            r_avg = np.mean(r_)
            bright_avg = np.mean(0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_)
            im = rgba_image_pil.convert("RGB")
            grad = getGradientMagnitude(np.array(im))
            unique, counts = np.unique(grad, return_counts=True)
            if counts[np.argwhere(unique <= 20)].sum() < PATCH_SIZE_revise * PATCH_SIZE_revise * 0.75:
                if bright_avg < 220 and r_avg > 120:
                    im_224 = im.resize((224, 224))
                    im_224.save(prefix_dir_224 + str(i) + '_' + str(j) + '.jpg')
        print(str(i/np.int64(width_whole/PATCH_SIZE_revise)*100)+'%')
    print('Finished WSI preprocessing part I')

    ###### 2. patching refine
    main_fp(args,root='./processed_data/'+args.dataset_name+'/',test_file_dir=root+files[file])
    print('Finished WSI preprocessing part II')

    ###### 3. read details
    generate_read_details_tiantan(root='./processed_data/'+args.dataset_name+'/')
    print('Finished WSI preprocessing part III')

    ###### 4. feature extraction
    feature_generation(root='./processed_data/'+args.dataset_name+'/')
    print('Finished WSI preprocessing part IV')
    #
    #






