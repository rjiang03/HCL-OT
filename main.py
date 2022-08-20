import argparse
import os
import pandas

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import ot
import utils
from model import Model
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))
global top
top = np.zeros((402,2))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def get_neg_mask(batch_size):
    postive_mask = torch.zeros((2 * batch_size, 2 * batch_size))
    for i in range(batch_size):
        postive_mask[i, i] = 1
        postive_mask[i, i + batch_size] = 1
        postive_mask[i + batch_size, i] = 1
        postive_mask[i + batch_size, i + batch_size] = 1
    negative_mask = 1-postive_mask
    return negative_mask.to(device)


def cost_fun(out_1, out_2):
    x = out_1[0].unsqueeze(0)
    y = out_2[0].unsqueeze(1)
    cost = torch.sum(torch.abs(x-y)**2,2)
    batch_size = out_1[0].shape[0]
    postive_mask = torch.zeros((batch_size, batch_size)).to(device)
    half_batch_size = int(batch_size/2)
    for i in range(half_batch_size):
        postive_mask[i, i] = float("Inf")
        postive_mask[i, i + half_batch_size] = float("Inf")
        postive_mask[i + half_batch_size, i] = float("Inf")
        postive_mask[i + half_batch_size, i + half_batch_size] = float("Inf")
    cost = cost + postive_mask
    return cost.reshape((1, cost.shape[0], cost.shape[1]))
    
def new_cost_fun(out_1, out_2, kappa):
    x = out_1[0].unsqueeze(0)
    y = out_2[0].unsqueeze(1)
    cost = torch.exp(torch.sum(torch.abs(x-y)**2,2)-kappa)
    batch_size = out_1[0].shape[0]
    postive_mask = torch.zeros((batch_size, batch_size)).to(device)
    half_batch_size = int(batch_size/2)
    for i in range(half_batch_size):
        postive_mask[i, i] = float("Inf")
        postive_mask[i, i + half_batch_size] = float("Inf")
        postive_mask[i + half_batch_size, i] = float("Inf")
        postive_mask[i + half_batch_size, i + half_batch_size] = float("Inf")
    cost = cost + postive_mask
    return cost.reshape((1, cost.shape[0], cost.shape[1]))
    
def OT_hard(out_1,out_2,batch_size, epoch, reg, tau_plus, kappa, new_cost):
        # neg score
        N = batch_size*2 - 2
        
        out = torch.cat([out_1, out_2], dim=0)
        X = out.cpu().detach().numpy()
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        neg_mask = get_neg_mask(batch_size)
        neg_masked = neg * neg_mask
        if reg > 0:
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)
            
            if new_cost:
                M = new_cost_fun([out], [out], kappa)
            else:
                M = cost_fun([out], [out])
                
            a = np.ones(batch_size*2)/(batch_size*2)
            Trans = ot.sinkhorn(a, a, M.cpu().detach().numpy()[0], reg)
            Trans = torch.tensor(Trans).cuda()   
            neg = neg_masked * Trans
            
            neg_2 = torch.sum(neg, 1)*batch_size*2*N
            Ng = (-tau_plus * N * pos + neg_2) / (1 - tau_plus)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        loss = (- torch.log(pos / (pos + Ng))).mean()
        return loss


def train(net, data_loader, train_optimizer, temperature, estimator, tau_plus, reg, kappa, new_cost):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        loss = OT_hard(out_1,out_2,batch_size, epoch, reg, tau_plus, kappa, new_cost)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, reg, dataset_name, estimator, tau):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            #print(epoch, epochs, total_num, total_top1 / total_num * 100, total_top5 / total_num * 100)
            top[epoch][0] = float(total_top1 / total_num * 100)
            top[epoch][1] = float(total_top5 / total_num * 100)
            print(epoch, epochs, 'total_top1', total_top1, total_num, total_top1 / total_num * 100, 'total_top5', total_top5, total_top5 / total_num * 100)
            #np.save(dataset_name+estimator+"reg"+str(reg)+"_unbalanced_"+str(reg_unbalance)+"tau"+str(tau)+".npy", top)
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
            

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='entropy_ot', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='cifar100', type=str, help='Choose loss function')
    parser.add_argument('--reg', default=0.7, type=float, help='entropy regularization')
    parser.add_argument('--new_cost', default=False, type=bool, help='Use default cost or not')
    parser.add_argument('--kappa', default=1, type=float, help='hyper parameter for new cost')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, estimator, reg = args.batch_size, args.epochs, args.estimator, args.reg
    dataset_name = args.dataset_name
    new_cost, kappa = args.new_cost, args.kappa

    

    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    print(dataset_name, '# Classes: {}'.format(c))
    

    print(dataset_name, "estimator", estimator, 'reg', reg, 'new_cost', new_cost, 'kappa', kappa)

    # training loop 
    if not os.path.exists('../results'):
        os.mkdir('../results')
    if not os.path.exists('../results/{}'.format(dataset_name)):
        os.mkdir('../results/{}'.format(dataset_name))
    for epoch in range(0, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature, estimator, tau_plus, reg, kappa, new_cost)
        
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, reg, dataset_name, estimator, tau_plus)
        torch.save(model.state_dict(), '../results/{}/{}_{}_{}_{}_{}_new.pth'.format(dataset_name,dataset_name,estimator,batch_size,tau_plus,reg))

