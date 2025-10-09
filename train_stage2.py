from utils import *
import dataset_mine
from net import init_weights, get_scheduler, WarmupCosineSchedule
from utils_finetune import *
from utils_server import *

def train(opt):
    gpuID = opt['gpus']
    ############### Mine_model #######################
    Mine_model_init,Mine_model_molecular,Mine_model_Graph, Mine_model_His, Mine_model_Cls\
        , opt_init,opt_molecular, opt_Graph,opt_His, opt_Cls = get_model_stage2(opt)

    if opt['decayType'] == 'exp' or opt['decayType'] == 'step':
        Mine_model_sch_molecular = get_scheduler(opt_molecular, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
        Mine_model_sch_Graph = get_scheduler(opt_Graph, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1)
    elif opt['decayType'] == 'cos':
        Mine_model_sch_molecular = WarmupCosineSchedule(opt_molecular, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])
        Mine_model_sch_Graph = WarmupCosineSchedule(opt_Graph, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))

    ###############  Datasets #######################

    ###############  Datasets #######################

    trainDataset = dataset_mine.Our_Dataset_stage2_3fold(phase='Train', opt=opt)
    testDataset = dataset_mine.Our_Dataset_stage2_3fold(phase='Test', opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0

    ckptdir = os.path.join(r'./models/'+opt['hispretrain']+'/Mine_model-0070.pth')
    checkpoint = torch.load(ckptdir)
    related_params = {k: v for k, v in checkpoint['init'].items()}
    Mine_model_init.load_state_dict(related_params, strict=True)
    related_params = {k: v for k, v in checkpoint['His'].items()}
    Mine_model_His.load_state_dict(related_params, strict=True)
    related_params = {k: v for k, v in checkpoint['Cls'].items()}
    Mine_model_Cls.load_state_dict(related_params, strict=True)
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()

    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep']

    ############# begin training ##########################
    for epoch in range(alleps):
        Mine_model_sch_molecular.step()
        Mine_model_sch_Graph.step()
        Mine_model_molecular.train()
        Mine_model_Graph.train()

        curep = last_ep + epoch
        lossdict = {'train/Molecular': 0, 'train/Graph': 0, 'train/DCC': 0, 'train/all': 0}
        count = 0
        count_IDH=0
        count_1p19q=0
        count_CDKN=0
        count_Diag=0
        count_Diag_sim=0
        running_results = {'acc_IDH': 0, 'acc_1p19q': 0, 'acc_CDKN': 0, 'acc_Diag_simple': 0
            , 'acc_Diag': 0, 'loss_Molecular': 0, 'loss_Graph': 0, 'loss_DCC': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img = packs[0]  ##(BS,N,1024)
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

            ### ### forward WHO 2007
            init_feature = Mine_model_init(img)  # (BS,2500,1024)
            hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,saliency_map_His,saliency_map_Grade)
            results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(encoded_His, encoded_Grade)
            pred_His = results_dict['logits_His']
            pred_Grade = results_dict['logits_Grade']


            ### ### forward molecular
            encoded_IDH,encoded_1p19q,encoded_CDKN = Mine_model_molecular(init_feature)
            results_dict, saliency_IDH_wt,saliency_1p19q_codel,encoded_IDH0,encoded_1p19q0,encoded_CDKN0 = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)
            pred_IDH = results_dict['logits_IDH']
            pred_1p19q = results_dict['logits_1p19q']
            pred_CDKN = results_dict['logits_CDKN']

            ### ### backward molecular
            Mine_model_molecular.zero_grad()
            Mine_model_Graph.zero_grad()

            loss_IDH = Mine_model_Graph.module.calculateLoss_IDH(pred_IDH,label_IDH)
            loss_1p19q = Mine_model_Graph.module.calculateLoss_1p19q(pred_1p19q, label_1p19q)
            loss_CDKN = Mine_model_Graph.module.calculateLoss_CDKN(pred_CDKN, label_CDKN)
            loss_Graph = Mine_model_Graph.module.calculateLoss_Graph(encoded_IDH0,encoded_1p19q0,encoded_CDKN0)

            if count % 10==0:
                for i in range(saliency_IDH_wt.detach().cpu().numpy().shape[0]):
                    if i==0:
                        loss_mutual_correlation =  Mine_model_Cls.module.Loss_mutual_correlation(
                            saliency_IDH_wt[i], saliency_GBM[i], saliency_1p19q_codel[i],saliency_O[i], epoch)
                    else:
                        loss_mutual_correlation+=Mine_model_Cls.module.Loss_mutual_correlation(
                            saliency_IDH_wt[i], saliency_GBM[i], saliency_1p19q_codel[i],saliency_O[i], epoch)
                loss_mutual_correlation=loss_mutual_correlation/(opt['batchSize'])
                loss_mutual_correlation.requires_grad_(True)
                loss_all = loss_IDH+loss_1p19q+loss_CDKN+opt['Network']['graph_loss_ratio']*loss_Graph+opt['Network']['corre_loss_ratio']*loss_mutual_correlation
                lossdict['train/DCC'] += loss_mutual_correlation.item()
            else:
                loss_all = loss_IDH + loss_1p19q + loss_CDKN + opt['Network']['graph_loss_ratio'] * loss_Graph
            loss_all.backward()
            opt_molecular.step()
            opt_Graph.step()

            ## prediction
            _, predicted_IDH = torch.max(pred_IDH.data, 1)
            correct_IDH,FLAT_normal,total_IDH,_,__ = cal_mole_correct(predicted_IDH,label_IDH)
            if FLAT_normal:
                running_results['acc_IDH'] += 100. * correct_IDH / total_IDH
                count_IDH+=1

            _, predicted_1p19q = torch.max(pred_1p19q.data, 1)
            correct_1p19q, FLAT_normal, total_1p19q ,_,__= cal_mole_correct(predicted_1p19q, label_1p19q)
            if FLAT_normal:
                running_results['acc_1p19q'] += 100. * correct_1p19q / total_1p19q
                count_1p19q += 1

            _, predicted_CDKN = torch.max(pred_CDKN.data, 1)
            correct_CDKN, FLAT_normal, total_CDKN ,_,__= cal_mole_correct(predicted_CDKN, label_CDKN)
            if FLAT_normal:
                running_results['acc_CDKN'] += 100. * correct_CDKN / total_CDKN
                count_CDKN += 1

            correct_Diag,FLAT_normal,total_Diag = cal_Diag_correct(opt,label_Diag, pred_IDH, pred_1p19q, pred_CDKN, pred_His,pred_Grade)
            if FLAT_normal:
                running_results['acc_Diag'] += 100. * correct_Diag / total_Diag
                count_Diag += 1

            correct_DiagSim, FLAT_normal, total_DiagSim = cal_DiagSim_correct(opt,label_Diag_simple, pred_IDH, pred_1p19q, pred_CDKN,pred_His)
            if FLAT_normal:
                running_results['acc_Diag_simple'] += 100. * correct_DiagSim / total_DiagSim
                count_Diag_sim += 1


            lossdict['train/Molecular'] += (loss_IDH+loss_1p19q+loss_CDKN).item()
            lossdict['train/Graph'] += loss_Graph.item()
            lossdict['train/all'] += loss_all.item()


            train_bar.set_description(
                desc=opt['name'] + '[%d/%d] l:%.1f|I:%.1f|1:%.1f|C:%.1f|D:%.1f|DS:%.1f' % (
                    epoch, alleps,
                    (lossdict['train/all'] / count_IDH),
                    (running_results['acc_IDH'] / count_IDH) if count_IDH else 0,
                    (running_results['acc_1p19q'] / count_1p19q) if count_1p19q else 0,
                    (running_results['acc_CDKN'] / count_CDKN) if count_CDKN else 0,
                    (running_results['acc_Diag'] / count_Diag) if count_Diag else 0,
                    (running_results['acc_Diag_simple'] / count_Diag_sim) if count_Diag_sim else 0,

                ))

        lossdict['train/Molecular'] = lossdict['train/Molecular'] / count_IDH
        lossdict['train/Graph'] = lossdict['train/Graph'] / count_IDH
        lossdict['train/DCC'] = lossdict['train/DCC'] / count_IDH
        lossdict['train/all'] = lossdict['train/all'] / count_IDH
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')



        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % 5 == 0:
            if (curep + 1) > (alleps / 2):
                save_dir = os.path.join(opt['modelDir'], 'Mine_model-%04d.pth' % (curep + 1))
                state = {
                    'init': Mine_model_init.state_dict(),
                    'His': Mine_model_His.state_dict(),
                    'Cls': Mine_model_Cls.state_dict(),
                    'molecular':Mine_model_molecular.state_dict(),
                    'Graph':Mine_model_Graph.state_dict(),
                }
                torch.save(state, save_dir)

            print("----------Test-------------")

            test_stage2_stem(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph,testLoader, gpuID,
                               epoch)

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
    parser.add_argument('--opt', type=str, default='config/mine_stage2.yml')
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
        train(opt)





























