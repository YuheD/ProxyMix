import argparse
# from email.policy import default
import os
import os.path as osp
import random
from unicodedata import digit

import numpy as np
# from numpy.core.fromnumeric import argmax
# from sklearn.utils.extmath import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import loss
import utils

@torch.no_grad()
def get_embedding(args, tgt_loader, model, cat_data=False, aug=False):
    model.eval()
    
    pred_bank = torch.zeros([len(tgt_loader.dataset), args.class_num]).cuda()
    emb_bank = torch.zeros([len(tgt_loader.dataset), args.bottleneck_dim]).cuda()

    for batch_idx, (data, target, idx) in enumerate(tgt_loader):
        data, target = data.cuda(), target.cuda()

        fea, out = model(data)
        emb_bank[idx] = fea
        pred_bank[idx] = out

    return pred_bank, emb_bank

def data_load_list(args, p_list, t_list):

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    source_set = utils.ObjectImage_list(p_list, train_transform)
    target_set = utils.ObjectImage_mul_list(t_list, [train_transform, train_transform])
    test_set = utils.ObjectImage("", args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["proxy"] = torch.utils.data.DataLoader(
        source_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
    )

    dset_loaders["target"] = torch.utils.data.DataLoader(
        target_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
    )

    dset_loaders["test"] = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )
    return dset_loaders

def list2txt(list, name):
    """save the list to txt file"""
    file = name     
    if os.path.exists(file):
        os.remove(file)
    for (path, label) in list:
        with open(file,'a+') as f:
            f.write(path+' '+ str(label)+'\n')

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

@torch.no_grad()
def data_split(args, base_network):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_set = utils.ObjectImage_mul('', args.t_dset_path, train_transform)
    split_loaders = {}
    split_loaders["split"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.worker, drop_last=False)

    NUMS = args.easynum  # easy samples of each class

    if(args.skip_split):
        filename_e = './data/{}/easy_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS) 
        filename_h = './data/{}/hard_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
        easy_path = utils.make_dataset('', filename_e)
        hard_path = utils.make_dataset('', filename_h)
        print('load txt from ' + filename_e + ' and ' + filename_h )             
        args.out_file.write('load txt from ' + filename_e + 'and' + filename_h  + '\n')             
        args.out_file.flush()
    else:
        easy_path, hard_path, easy_idx, hard_idx = [], [], [], []

        base_network.eval()
        """ the full (path, label) list """
        img = utils.make_dataset('', args.t_dset_path)

        # with torch.no_grad():
        """ extract the prototypes """
        for name, param in base_network.named_parameters():
            if('fc.weight' in name):
                prototype = param

        _, features_bank = get_embedding(args, split_loaders["split"], base_network)
        features_bank = F.normalize(features_bank) # len * 256
        prototype = F.normalize(prototype) # cls * 256
        dists = prototype.mm(features_bank.t())  # cls * len

        sort_idxs = torch.argsort(dists, dim=1, descending=True) #cls * len
        fault = 0.

        for i in range(args.class_num):
            ## check if the repeated index in the list
            s_idx = 0
            for _ in range(NUMS):
                idx = sort_idxs[i, s_idx]

                while idx in easy_idx:
                    s_idx += 1
                    idx = sort_idxs[i, s_idx]

                assert idx not in easy_idx

                easy_idx.append(idx)
                easy_path.append((img[idx][0], i))

                if not img[idx][1] == i:
                    fault += 1
                s_idx += 1


        for id in range(len(img)):
            if id not in easy_idx:
                hard_path.append(img[id])
                hard_idx.append(id)

        """mindist: a distance matrix which store the minimum cosine distance between the prototypes and features """

        acc = 1 - fault / (args.class_num*NUMS)

        print('Splited data list label Acc:{}'.format(acc))
        args.out_file.write('Splited data list label Acc:{}'.format(acc) + '\n')
        args.out_file.flush()

        if args.save_proxy:
            filename_e = './data/{}/easy_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
            filename_h = './data/{}/hard_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
            list2txt(easy_path, filename_e)
            list2txt(hard_path, filename_h)
            print('Splited data list saved in ' + filename_e + ' and ' + filename_h )
            args.out_file.write('Splited data list saved in ' + filename_e + 'and' + filename_h  + '\n')
            args.out_file.flush()

    return  easy_path, hard_path

def KLD(sfm, sft):
    return -torch.mean(torch.sum(sfm.log() * sft, dim=1))

def train(args):
    ## set pre-process

    class_num = args.class_num
    class_weight_src = torch.ones(class_num, ).cuda()

    ##################################################################################################

    ## set base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()  
    elif args.net == 'resnet34':
        netG = utils.ResBase34().cuda()  

    netF = utils.ResClassifier(class_num=class_num, feature_dim=netG.in_features, 
        bottleneck_dim=args.bottleneck_dim, type = args.cls_type, ltype = args.layer_type).cuda()
    base_network = nn.Sequential(netG, netF)

    for name, param in base_network.named_parameters():
        if('fc' in name):
            param.requires_grad = False

    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr )
    optimizer_f = optim.SGD(netF.parameters(), lr = args.botlr )

    for name, param in netF.named_parameters():
        if not param.requires_grad:
            print(name + ' fixed.')
            args.out_file.write(name + ' fixed.' + '\n')
            args.out_file.flush()

    base_network.load_state_dict(torch.load(args.ckpt))
    print('load source model from '+ args.ckpt + '\n')
    args.out_file.write('load source model from '+ args.ckpt + '\n')
    args.out_file.flush()

    """split the target set"""

    easy_path, hard_path = data_split(args, base_network)

    ## set dataloaders

    dset_loaders = data_load_list(args, easy_path, easy_path + hard_path)

    max_len =  len(dset_loaders["target"])
    args.max_iter = args.max_epoch * max_len
    # eval_iter = args.val_num

    ## Memory Bank
    if args.pl.endswith('na'):
        mem_fea = torch.rand(len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num

    proxy_loader_iter = iter(dset_loaders["proxy"])
    target_loader_iter = iter(dset_loaders["target"])

    list_acc = []
    best_ent = 100

    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr , iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.botlr  , iter_num=iter_num, max_iter=args.max_iter)

        try:
            inputs_proxy, labels_proxy = proxy_loader_iter.next()
        except:
            proxy_loader_iter = iter(dset_loaders["proxy"])
            inputs_proxy, labels_proxy = proxy_loader_iter.next()
        inputs_proxy, labels_proxy = inputs_proxy.cuda(),  labels_proxy.cuda()

        try:
            inputs_target_all, _, idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target_all, _, idx = target_loader_iter.next()

        inputs_target = inputs_target_all[0].cuda()
        inputs_target_aug = inputs_target_all[1].cuda()

        features_target, outputs_target = base_network(inputs_target)

        total_loss = torch.tensor(0.).cuda()

        eff = iter_num / args.max_iter

        if args.src_ratio:
            _, outputs_source = base_network(inputs_proxy)

            src_ = loss.CrossEntropyLabelSmooth(reduction=None,num_classes=class_num, epsilon=args.smooth)(outputs_source, labels_proxy)

            weight_src = class_weight_src[labels_proxy].unsqueeze(0)
            src_cls = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())

            total_loss += src_cls * args.src_ratio

        """ hard sample soft label generation """

        softmax_out = nn.Softmax(dim=1)(outputs_target)

        if args.pl.endswith('na'):
            dis = -torch.mm(features_target.detach(), mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, idx[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)

            w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
            for wi in range(w.size(0)):
                for wj in range(args.K):
                    w[wi][p1[wi, wj]] = 1/ args.K

            sft_label = w.mm(mem_cls)

        else:
            raise RuntimeError('pseudo label error')


        """ mixup easy and hard samples """

        mix_cls_loss = torch.tensor(0.).cuda()

        if args.mix_ratio:
            rho = np.random.beta(args.alpha, args.alpha)
            
            mix_img = inputs_target * rho + inputs_proxy*(1-rho)

            _, mix_out = base_network(mix_img)
            weight_src = class_weight_src[labels_proxy].unsqueeze(0)

            targets_s = torch.zeros(args.batch_size, args.class_num).cuda().scatter_(1, labels_proxy.view(-1,1), 1)
            mix_target = sft_label * rho + targets_s * (1-rho)
            mix_cls_loss += eff * KLD(nn.Softmax(dim=1)(mix_out), mix_target)

        """ hard samples regression loss """

        remix_reg_loss = torch.tensor(0.).cuda()

        if args.reg_ratio:
            inputs_t = inputs_target
            inputs_t2 = inputs_target_aug

            targets_u = 0.5 * sft_label
            if args.pl.endswith('na'):
                # print(1)
                with torch.no_grad():
                    features_target, outputs_u = base_network(inputs_t2)
                    dis = -torch.mm(features_target.detach(), mem_fea.t())
                    for di in range(dis.size(0)):
                        dis[di, idx[di]] = torch.max(dis)
                    _, p1 = torch.sort(dis, dim=1)
                    w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
                    for wi in range(w.size(0)):
                        for wj in range(args.K):
                            w[wi][p1[wi, wj]] = 1/ args.K
                    sft_label2 = w.mm(mem_cls)

            targets_u += 0.5 * sft_label2

            rho = np.random.beta(args.alpha, args.alpha)

            all_inputs = torch.cat([inputs_t, inputs_t2], dim=0)
            all_targets = torch.cat([targets_u, targets_u], dim=0)
            ind = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[ind]
            target_a, target_b = all_targets, all_targets[ind]
            mixed_input = rho * input_a + (1 - rho) * input_b
            mixed_target = rho * target_a + (1 - rho) * target_b

            """interleave labeled and unlabed samples between batches to get correct batchnorm calculation"""
            mixed_input = list(torch.split(mixed_input, args.batch_size))
            mixed_input = utils.interleave(mixed_input, args.batch_size)
            features, logits = base_network(mixed_input[0])
            logits = [logits]
            for input in mixed_input[1:]:
                _, temp = base_network(input)
                logits.append(temp)

            # put interleaved samples back
            logits = utils.interleave(logits, args.batch_size)
            logits_u = torch.cat(logits[0:], dim=0)

            probs_u = torch.softmax(logits_u, dim=1) 
            Lu = (torch.mean((probs_u - mixed_target) ** 2) * args.lambda_u * utils.linear_rampup(iter_num, args.max_iter))

            remix_reg_loss += Lu 

        total_loss += mix_cls_loss * args.mix_ratio + remix_reg_loss * args.reg_ratio

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        
        """ update the memory bank """
        if args.pl.endswith('na'):
            base_network.eval() 
            with torch.no_grad():
                features_target, outputs_target = base_network(inputs_target)
                features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
                softmax_out = nn.Softmax(dim=1)(outputs_target)

                if args.pl == 'fw_na':
                    # print(3)
                    sfm_out = softmax_out ** 2 / softmax_out.sum(dim=0)
                    outputs_target = sfm_out / sfm_out.sum(dim=1, keepdim=True)

                else:
                    raise RuntimeError('pseudo label error')

            mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
            mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()

        if iter_num % int(args.eval_epoch*max_len) == 0:
            base_network.eval()
            if args.dset == 'VISDA-C':
                acc, py, score, y, tacc = utils.cal_acc_visda(dset_loaders["test"], base_network)
                args.out_file.write(tacc + '\n')
                args.out_file.flush()

                _ent = loss.Entropy(score)
                mean_ent = 0
                for ci in range(args.class_num):
                    mean_ent += _ent[py==ci].mean()
                mean_ent /= args.class_num

            else:
                acc, py, score, y = utils.cal_acc(dset_loaders["test"], base_network)
                mean_ent = torch.mean(loss.Entropy(score))

            list_acc.append(acc * 100)
            if best_ent > mean_ent:
                best_ent = mean_ent
                val_acc = acc * 100
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, iter_num, args.max_iter, acc*100, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
    
    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}\nMax Acc = {:.2f}\nFin Acc = {:.2f}\n'.format(val_acc, max_acc, final_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    torch.save(base_network.state_dict(), osp.join(args.output_dir, args.log + ".pt"))

    return best_y.cpu().numpy().astype(np.int64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly')
    parser.add_argument('--dset', type=str, default='office-home', choices=['IMAGECLEF', 'VISDA-C', 'office', 'office-home','DomainNet126'], help="The dataset or source dataset used")

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--tname', type=str, default=None, help="target")
    parser.add_argument('--Nofinetune', action='store_true')
    parser.add_argument('--test_on_src', action='store_true')
    parser.add_argument('--pl', type=str, default='fw_na',choices=['mixmatch', 'fw', 'mixmatch_na','remixmatch_na','fw_na','atdoc_na'])
    parser.add_argument('--split', type=str, default='proto',choices=['proto', 'ent','rand'])

    parser.add_argument('--mix_ratio', type=float, default=1)
    parser.add_argument('--reg_ratio', type=float, default=1)
    parser.add_argument('--src_ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.75)

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_epoch', type=str, default=None)
    parser.add_argument('--output', type=str, default='tmp/',required=True)

    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")

    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument("--lambda_u", default=100, type=float)

    parser.add_argument('--easynum', type=int, default=10) 
    parser.add_argument('--skip_split', action='store_true')
    parser.add_argument('--save_proxy', action='store_true')
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101",'resnet34'])
    parser.add_argument('--cls_type', type=str, default='ori')
    parser.add_argument('--layer_type', type=str, default='linear')
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--botlr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--momentum', type=float, default=1)

    args = parser.parse_args()
    args.output = 'logs/'+ args.output
    args.output = args.output.strip()

    args.momentum = 1.0

    args.eval_epoch = args.max_epoch / 5

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'DomainNet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'IMAGECLERF':
        names = ['c','i','p']
        args.class_num = 12
    # if args.dset == 'digit':
    #     names = []
    #     args.class_num = 

    if(args.ckpt is None):
        if(args.ckpt_epoch is not None):
            args.ckpt = './logs/source_' + args.ckpt_epoch + '/' + args.dset + '/' + names[args.s][0].upper() + names[args.s][0].upper() + '/srconly.pt' 
        else:
            args.ckpt = './logs/source/' + args.dset + '/' + names[args.s][0].upper() + names[args.s][0].upper() + '/srconly.pt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.s_dset_path = './data/' + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_list.txt'

    if(args.tname is not None):
        args.t_dset_path = './data/' + args.dset + '/' + args.tname + '_list.txt'
    args.test_dset_path = args.t_dset_path
    args.sname = names[args.s]

    if(args.test_on_src):
        args.test_dset_path = args.s_dset_path
        args.t_dset_path = args.s_dset_path

    args.output_dir = osp.join(args.output, 'pl_' + str(args.pl) + '_src_' + str(args.src_ratio) + '_mix_' + str(args.mix_ratio) + '_reg_' + str(args.reg_ratio) + '_split_' + str(args.split),
                               args.dset, names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.log = args.method
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    utils.print_args(args)
    
    label = train(args)
    
