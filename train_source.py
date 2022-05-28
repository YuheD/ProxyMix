import argparse
import os
import os.path as osp
import random

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import loss
import utils

def data_load(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_set = utils.ObjectImage('', args.s_dset_path, train_transform)
    test_set = utils.ObjectImage('', args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def train(args, label=None):
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

    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr = args.lr )


    ## set dataloaders

    dset_loaders = data_load(args)

    max_len = len(dset_loaders["source"])
    args.max_iter = args.max_epoch * max_len
        
    source_loader_iter = iter(dset_loaders["source"])

    list_acc = []
    best_ent = 100

    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr , iter_num=iter_num, max_iter=args.max_iter)

        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()       
        inputs_source, labels_source = inputs_source.cuda(),  labels_source.cuda()
        _, outputs_source = base_network(inputs_source)

        src_ = loss.CrossEntropyLabelSmooth(reduction=False,num_classes=class_num, epsilon=args.smooth)(outputs_source, labels_source)
        weight_src = class_weight_src[labels_source].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        
        total_loss =  classifier_loss

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        
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

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")

    parser.add_argument('--ckpt', type=str, default=None)
    
    parser.add_argument('--output', type=str, default='source')
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--smooth', type=float, default=0.1)
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101", "resnet34"])
    parser.add_argument('--cls_type', type=str, default='ori')
    parser.add_argument('--layer_type', type=str, default='linear')
    parser.add_argument('--dset', type=str, default='office-home', choices=[
                        'IMAGECLERF', 'VISDA-C', 'office', 'office-home', 'DomainNet126'], help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    args = parser.parse_args()
    args.output = 'logs/'+ args.output
    args.output = args.output.strip()

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
    if (args.dset == 'IMAGECLERF'):
        names = ['c','i','p']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.s_dset_path = './data/' + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = args.s_dset_path

    args.output_dir = osp.join(args.output, args.dset, 
        names[args.s][0].upper() + names[args.s][0].upper())

    args.name = names[args.s][0].upper() + names[args.s][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.log = 'srconly'
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    utils.print_args(args)
    
    label = train(args)

