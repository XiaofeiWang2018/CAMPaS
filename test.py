import os.path

from utils import get_model_endtoend,get_model_endtoendCNN,saliency_predcls_gene,saliency_predclsSim_gene
from utils_server import *
from yaml.loader import SafeLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = 250000000000
import platform
from dataset_mine import *
sysstr = platform.system()
import warnings
warnings.filterwarnings("ignore")
import dataset_mine

def metrics_generation(opt):
    gpuID = opt['gpus']

    Mine_model_init, Mine_model_molecular, Mine_model_Graph, Mine_model_His, Mine_model_Cls , opt_init, opt_molecular, opt_Graph, opt_His, opt_Cls = get_model_stage2(opt)
    ###############  Datasets #######################
    testDataset = dataset_mine.Our_Dataset_stage2_3fold(phase='Test', opt=opt)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],num_workers=8, shuffle=False)
        ############## initialize #######################
    root = './models/' + opt['test_pth']
    ckptdir = os.path.join(root)
    checkpoint = torch.load(ckptdir, map_location={'cuda:0': 'cuda:'+str(opt['gpus'][0])})
    related_params = {k: v for k, v in checkpoint['init'].items()}
    Mine_model_init.load_state_dict(related_params, strict=True)
    related_params = {k: v for k, v in checkpoint['His'].items()}
    Mine_model_His.load_state_dict(related_params, strict=True)
    related_params = {k: v for k, v in checkpoint['Cls'].items()}
    Mine_model_Cls.load_state_dict(related_params, strict=True)
    related_params = {k: v for k, v in checkpoint['molecular'].items()}
    Mine_model_molecular.load_state_dict(related_params, strict=True)
    related_params = {k: v for k, v in checkpoint['Graph'].items()}
    Mine_model_Graph.load_state_dict(related_params, strict=True)
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    Mine_model_molecular.eval()
    Mine_model_Graph.eval()
    print("-----`-----Test-------------")
    if opt['TrainingSet'] == 'TCGA':
        test_stage2_stem(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, Mine_model_molecular, Mine_model_Graph,testLoader, gpuID)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/test.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)
    setup_seed(opt['seed'])

    metrics_generation(opt)# model='ours''ViT''CNN''CLAM''TransMIL'


