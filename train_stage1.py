
from utils import *
import dataset_mine
from utils_server import *
from utils_finetune import *
from net import init_weights, get_scheduler, WarmupCosineSchedule

def find_label_noise(opt):
    if not os.path.exists( './noise'):
        os.makedirs('./noise')
    find_label_noise_onefold(opt,fold=0)
    find_label_noise_onefold(opt, fold=1)
    find_label_noise_onefold(opt, fold=2)

def find_label_noise_onefold(opt,fold):

    excel_label_wsi = pd.read_excel(opt['label_path'] + 'TCGA.xlsx', sheet_name='wsi_level', header=0)
    excel_wsi = excel_label_wsi.values
    PATIENT_LIST = excel_wsi[:, 0]
    np.random.seed(opt['seed'])
    random.seed(opt['seed'])
    PATIENT_LIST = list(PATIENT_LIST)
    PATIENT_LIST = np.unique(PATIENT_LIST)
    np.random.shuffle(PATIENT_LIST)
    NUM_PATIENT_ALL = len(PATIENT_LIST)
    if fold == 0:
        TRAIN_PATIENT_LIST_TCGA = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.67)]
    elif fold == 1:
        TRAIN_PATIENT_LIST_TCGA = np.concatenate(
            (PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.33)], PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]))
    elif fold == 2:
        TRAIN_PATIENT_LIST_TCGA = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.33):]



    TRAIN_LIST = []
    for i in range(excel_wsi.shape[0]):  # 2612
        if excel_wsi[:, 0][i] in TRAIN_PATIENT_LIST_TCGA:
            TRAIN_LIST.append(excel_wsi[i, :])

    LIST = np.asarray(TRAIN_LIST)
    clean_subtype, noisy_subtype, clean_grade, noisy_grade = count_noisy_clean(LIST)
    with open('noise/fold' + str(fold) + '/Grade_Train_noise.txt', 'w+') as f:
        for i in range(noisy_grade.shape[0]):
            f.write(noisy_grade[i, 1])
            f.write('\n')
            f.flush()
    with open('noise/fold' + str(fold) + '/His_Train_noise.txt', 'w+') as f:
        for i in range(noisy_subtype.shape[0]):
            f.write(noisy_subtype[i, 1])
            f.write('\n')
            f.flush()
    a = 1

def train_pre_salien(opt):

    gpuID = opt['gpus']
    ############### Mine_model #######################
    Mine_model_init,Mine_model_His,Mine_model_Cls,opt_init,opt_His,opt_Cls=get_model_stage1(opt)

    if opt['decayType']=='exp' or opt['decayType']=='step':
        Mine_model_sch_init = get_scheduler(opt_init, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_His = get_scheduler(opt_His, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_Cls = get_scheduler(opt_Cls, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
    elif opt['decayType']=='cos':
        Mine_model_sch_init = WarmupCosineSchedule(opt_init, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_His = WarmupCosineSchedule(opt_His, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
        Mine_model_sch_Cls = WarmupCosineSchedule(opt_Cls, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))

    ###############  Datasets #######################

    trainDataset = dataset_mine.Our_Dataset_stage1_3fold(phase='Train', opt=opt)
    testDataset = dataset_mine.Our_Dataset_stage1_3fold(phase='Test', opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
    trainLoader_gene = DataLoader(trainDataset, batch_size=1,
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep_pre']


    ############# begin training ##########################
    for epoch in range(alleps):
        Mine_model_sch_init.step()
        Mine_model_sch_His.step()
        Mine_model_sch_Cls.step()
        Mine_model_init.train()
        Mine_model_His.train()
        Mine_model_Cls.train()
        curep = last_ep + epoch
        lossdict = { 'train/His': 0,'train/Grade': 0,'train/His_all': 0}
        count=0
        count_his=0
        running_results = {'acc_His': 0,'acc_Grade': 0
                           ,'loss_His': 0,'loss_Grade ': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img = packs[0] ##(BS,N,1024)
            label = packs[1]
            file_name = packs[2]

            count+=1
            count_his+=1
            if  torch.cuda.is_available():
                img = img.cuda(gpuID[0])
                label = label.cuda(gpuID[0])
            label_his=label[:,0]
            label_Grade = label[:, 1]

            ### ### generate re-weighting paras
            imp_his, imp_grade=imp_gene(opt,img)
            ### ### forward His
            init_feature = Mine_model_init(img)  # (BS,2500,1024)
            hidden_states_his, hidden_states_grade,encoded_His,encoded_Grade = Mine_model_His(init_feature,imp_his, imp_grade)
            results_dict,saliency_A,saliency_O,saliency_GBM,saliency_G2,saliency_G3,saliency_G4 = Mine_model_Cls(encoded_His, encoded_Grade)
            pred_His = results_dict['logits_His']
            pred_Grade = results_dict['logits_Grade']



            ### ### backward His
            Mine_model_Cls.zero_grad()
            Mine_model_His.zero_grad()
            Mine_model_init.zero_grad()
            loss_His = Mine_model_His.module.calculateLoss_His_ori(pred_His, label_his)
            loss_Grade = Mine_model_His.module.calculateLoss_Grade_ori(pred_Grade, label_Grade)


            loss_His_all =loss_Grade+loss_His
            loss_His_all.backward()
            opt_init.step()
            opt_His.step()
            opt_Cls.step()


            _, predicted_His = torch.max(pred_His.data, 1)
            total_His = label_his.size(0)-list(label_his.detach().cpu().numpy()).count(3)
            correct_His = predicted_His.eq(label_his.data).cpu().sum()

            _, predicted_Gade = torch.max(pred_Grade.data, 1)
            total_Grade = label_Grade.size(0)
            correct_Grade = predicted_Gade.eq(label_Grade.data).cpu().sum()
            if total_His:
                running_results['acc_His'] += 100. * correct_His / total_His
            else:
                count_his-=1

            running_results['acc_Grade'] += 100. * correct_Grade / total_Grade


            lossdict['train/His'] += loss_His.item()
            lossdict['train/Grade'] += loss_Grade.item()
            lossdict['train/His_all'] += loss_His_all.item()

            train_bar.set_description(
                desc='Pre-'+opt['name'] + ' [%d/%d] l_H:%.2f |l_G:%.2f |H:%.2f |G:%.2f' % (
                    epoch, alleps,
                    lossdict['train/His'] / count,
                    lossdict['train/Grade'] / count,
                    running_results['acc_His'] / count,
                    running_results['acc_Grade'] / count,

                ))

        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % 5 == 0:
            save_dir = os.path.join(opt['modelDir'], 'Pre_Mine_model-%04d.pth' % (curep + 1))
            state = {
                'init': Mine_model_init.state_dict(),
                'His': Mine_model_His.state_dict(),
                'Cls': Mine_model_Cls.state_dict(),
            }
            torch.save(state, save_dir)




def train(opt):
    gpuID = opt['gpus']
    ############### Mine_model #######################
    Mine_model_init, Mine_model_His, Mine_model_Cls, opt_init, opt_His, opt_Cls = get_model_stage1(opt)

    if opt['decayType'] == 'exp' or opt['decayType'] == 'step':
        Mine_model_sch_init = get_scheduler(opt_init, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_His = get_scheduler(opt_His, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_Cls = get_scheduler(opt_Cls, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
    elif opt['decayType'] == 'cos':
        Mine_model_sch_init = WarmupCosineSchedule(opt_init, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])
        Mine_model_sch_His = WarmupCosineSchedule(opt_His, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])
        Mine_model_sch_Cls = WarmupCosineSchedule(opt_Cls, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))

    ###############  Datasets #######################

    trainDataset = dataset_mine.Our_Dataset_stage1_3fold(phase='Train', opt=opt)
    testDataset = dataset_mine.Our_Dataset_stage1_3fold(phase='Test', opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0

    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep']
    with open('./noise/fold'+str(opt['fold'])+'/His_Train_noise.txt') as f:
        His_Train_noise_list = [line.rstrip() for line in f]
    with open('./noise/fold'+str(opt['fold'])+'/Grade_Train_noise.txt') as f:
        Grade_Train_noise_list = [line.rstrip() for line in f]

    ############# begin training ##########################
    for epoch in range(alleps):
        Mine_model_sch_init.step()
        Mine_model_sch_His.step()
        Mine_model_sch_Cls.step()
        Mine_model_init.train()
        Mine_model_His.train()
        Mine_model_Cls.train()
        curep = last_ep + epoch
        lossdict = {'train/His': 0, 'train/Grade': 0, 'train/all': 0}
        count_his = 0
        running_results = {'acc_His': 0, 'acc_Grade': 0
            , 'loss_His': 0, 'loss_Grade ': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img = packs[0]  ##(BS,N,1024)
            label = packs[1]
            file_name = packs[2]


            if torch.cuda.is_available():
                img = img.cuda(gpuID[0])
                label = label.cuda(gpuID[0])
            label_his = label[:, 0]
            label_Grade = label[:, 1]

            saliency_map_His, saliency_map_Grade = saliency_map_read(opt, file_name, epoch)
            saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
            saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])

            ### ### forward His
            init_feature = Mine_model_init(img)  # (BS,2500,1024)
            hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,
                                                                                                saliency_map_His,
                                                                                                saliency_map_Grade)
            results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
                encoded_His, encoded_Grade)
            pred_His = results_dict['logits_His']
            pred_Grade = results_dict['logits_Grade']

            ### ### backward His
            Mine_model_Cls.zero_grad()
            Mine_model_His.zero_grad()
            Mine_model_init.zero_grad()

            loss_His = Mine_model_His.module.calculateLoss_His(pred_His, label_his,file_name,His_Train_noise_list)
            loss_Grade = Mine_model_His.module.calculateLoss_Grade(pred_Grade, label_Grade,file_name,Grade_Train_noise_list)

            loss_His_all = loss_Grade + loss_His
            loss_His_all.backward()
            opt_init.step()
            opt_His.step()
            opt_Cls.step()

            _, predicted_His = torch.max(pred_His.data, 1)
            total_His = label_his.size(0) - list(label_his.detach().cpu().numpy()).count(3)
            correct_His = predicted_His.eq(label_his.data).cpu().sum()

            _, predicted_Gade = torch.max(pred_Grade.data, 1)
            total_Grade = label_Grade.size(0)
            correct_Grade = predicted_Gade.eq(label_Grade.data).cpu().sum()
            if total_His:
                count_his += 1
                running_results['acc_His'] += 100. * correct_His / total_His
            else:
                count_his -= 1

            running_results['acc_Grade'] += 100. * correct_Grade / total_Grade

            lossdict['train/His'] += loss_His.item()
            lossdict['train/Grade'] += loss_Grade.item()
            lossdict['train/all'] += loss_His_all.item()

            train_bar.set_description(
                desc=opt['name'] + ' [%d/%d] l_H:%.2f |l_G:%.2f |H:%.2f |G:%.2f' % (
                    epoch, alleps,
                    lossdict['train/His'] / count_his,
                    lossdict['train/Grade'] / count_his,
                    running_results['acc_His'] / count_his,
                    running_results['acc_Grade'] / count_his,

                ))

        lossdict['train/His'] = lossdict['train/His'] / count_his
        lossdict['train/Grade'] = lossdict['train/Grade'] / count_his
        lossdict['train/all'] = lossdict['train/all'] / count_his
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')

        assert len(opt['saliency_ep']) == 2
        if epoch == opt['saliency_ep'][0] or epoch == opt['saliency_ep'][1]:
            print("----------generate_saliency-------------")
            generate_saliency(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, trainDataset, testLoader,gpuID,epoch)

        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % 5 == 0:
            if (curep + 1) > (alleps / 2):
                save_dir = os.path.join(opt['modelDir'], 'Mine_model-%04d.pth' % (curep + 1))
                state = {
                    'init': Mine_model_init.state_dict(),
                    'His': Mine_model_His.state_dict(),
                    'Cls': Mine_model_Cls.state_dict(),
                }
                torch.save(state, save_dir)

            test_stage1_stem(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, testLoader, gpuID,epoch)



def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


def remove_all_dir(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            for j in os.listdir(path_file):
                path_file1 = os.path.join(path_file, j)
                os.remove(path_file1)
            os.rmdir(path_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/mine_stage1.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)

    import platform

    sysstr = platform.system()

    setup_seed(opt['seed'])
    if opt['command'] == 'Train':
        cur_time = time.strftime('%m%d-%H%M', time.localtime())
        opt['name'] = opt['name'] + '-fold' + str(opt['fold']) + '-' + opt['TrainingSet']
        opt['name'] = opt['name'] + '_{}'.format(cur_time)
        opt['logDir'] = os.path.join(opt['logDir'], opt['name'])
        opt['modelDir'] = os.path.join(opt['modelDir'], opt['name'])
        opt['saveDir'] = os.path.join(opt['saveDir'], opt['name'])
        opt['cm_saveDir'] = os.path.join(opt['cm_saveDir'], opt['name'])
        if not os.path.exists(opt['logDir']):
            os.makedirs(opt['logDir'])
        if not os.path.exists(opt['modelDir']):
            os.makedirs(opt['modelDir'])
        if not os.path.exists(opt['saveDir']):
            os.makedirs(opt['saveDir'])
        if not os.path.exists(opt['cm_saveDir']):
            os.makedirs(opt['cm_saveDir'])
        para_log = os.path.join(opt['modelDir'], 'params.yml')
        if os.path.exists(para_log):
            os.remove(para_log)
        with open(para_log, 'w') as f:
            data = yaml.dump(opt, f, sort_keys=False, default_flow_style=False)

        print("\n\n============> begin training <=======")
        find_label_noise(opt)
        train_pre_salien(opt)
        train(opt)

    a = 1





























