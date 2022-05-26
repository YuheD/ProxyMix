import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class ResBase34(nn.Module):
    def __init__(self):
        super(ResBase34, self).__init__()
        model_resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase50(nn.Module):
    def __init__(self, center=None):
        super(ResBase50, self).__init__()
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.in_features = model_resnet50.fc.in_features
        if center is not None:
            self.center = torch.nn.Parameter(center)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_feature(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        a = self.layer1(x)
        f.append(a)
        b = self.layer2(a)
        f.append(b)
        c = self.layer3(b)
        f.append(c)
        d = self.layer4(c)
        f.append(d)
        return f

class ResBase101(nn.Module):
    def __init__(self):
        super(ResBase101, self).__init__()
        model_resnet101 = torchvision.models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_feature(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        a = self.layer1(x)
        f.append(a)
        b = self.layer2(a)
        f.append(b)
        c = self.layer3(b)
        f.append(c)
        d = self.layer4(c)
        f.append(d)
        return f

class ResClassifier(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=256, type = 'bn', ltype = 'wn'):
        super(ResClassifier, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type
        if(self.type=='bn'):
            self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        if(self.type == 'wn'):
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        y = self.fc(x)
        return x, y

class Classifierwobot(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=256):
        super(Classifierwobot, self).__init__()
        self.fc = nn.Linear(feature_dim, class_num)
        self.fc.apply(init_weights)

    def forward(self, x, skip_bot=False):
        # print(feature)
        y = self.fc(x)
        return x, y

class Feature_net(nn.Module):
    def __init__(self, net, bottleneck_dim=256):
        super(Feature_net, self).__init__()
        if net == "resnet101":
            self.resnet = ResBase101()
        else:
            self.resnet = ResBase50()
        self.bottleneck = nn.Linear(self.resnet.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.resnet(x)
        x = self.bottleneck(x)
        return x

    def get_feature(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        a = self.layer1(x)
        f.append(a)
        b = self.layer2(a)
        f.append(b)
        c = self.layer3(b)
        f.append(c)
        d = self.layer4(c)
        f.append(d)
        return f

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert("RGB")

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(" ")
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images

def make_cls_dataset(root, label):
    images = []
    tag = []
    labeltxt = open(label)
    last_gt = 0
    i = 0
    for line in labeltxt:
        data = line.strip().split(" ")
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        if gt != last_gt:
            tag.append(i)
        i += 1
        last_gt = gt
        item = (path, gt)
        images.append(item)
        # print(tag,len(tag))
    return images, tag[1] - tag[0]

class ObjectImage_y(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, y=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.y = y

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        target = self.y[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # path = '../../../DATA' + path[9:]
        # print(path)
        # raise RuntimeError
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ObjectImage_list(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None, loader=default_loader):
        self.imgs = data_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ObjectImage_mul(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # path = '../../../DATA' + path[9:]
        img = self.loader(path)
        if self.transform is not None:
            # print(type(self.transform).__name__)
            if type(self.transform).__name__ == "list":
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ObjectImage_mul_list(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None, loader=default_loader):
        self.imgs = data_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == "list":
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

def print_args(args):
    log_str = "==========================================\n"
    log_str += "==========       config      =============\n"
    log_str += "==========================================\n"
    for arg, content in args.__dict__.items():
        log_str += "{}:{}\n".format(arg, content)
    log_str += "\n==========================================\n"
    print(log_str)
    args.out_file.write(log_str + "\n")
    args.out_file.flush()

def cal_fea(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels = iter_test.next()
            inputs = inputs.cuda()
            feas, outputs = model(inputs)
            if start_test:
                all_feas = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feas = torch.cat((all_feas, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_feas, all_label

def cal_acc(loader, model, flag=True, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    return accuracy, predict, all_output, all_label

def cal_acc2(loader, model, model2, flag=True, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
                _, outputs2 = model2(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                    feas2, outputs2 = model2(inputs)
                    outputs2 = fc(feas2)
                else:
                    outputs = model(inputs)
                    outputs2 = model2(inputs)
            # outputs = (outputs + outputs2)/2.

            if start_test:
                all_output = outputs.float().cpu()
                all_output2 = outputs2.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_output2 = torch.cat((all_output2, outputs2.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    all_output2 = nn.Softmax(dim=1)(all_output2)

    all_output = (all_output2 + all_output) / 2.0
    _, predict = torch.max(all_output, 1)

    # print(predict.max())
    # predict =predict % 65
    # print(predict.max())
    # raise RuntimeError
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    return accuracy, predict, all_output, all_label

def cal_acc_visda(loader, model, flag=True, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean() / 100
    aa = [str(np.round(i, 2)) for i in acc]
    acc = " ".join(aa)
    print(acc)

    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return aacc, predict, all_output, all_label, acc

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(
        self,
        outputs_x,
        targets_x,
        outputs_u,
        targets_u,
        epoch,
        max_epochs=30,
        lambda_u=75,
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]