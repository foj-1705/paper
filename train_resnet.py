from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


#from fgsm import FGSM

#from autoattack import AutoAttack

from wideresnet import *
from resnet import *
from snart import lrat_loss
import numpy as np
import time


parser = argparse.ArgumentParser(description='Robustness through Cross-Entropy Ratio')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=0.4,
                    help='regularization parameter')
parser.add_argument('--training_version', default='lrat',
                    help = 'select lrat or lrllat')

parser.add_argument('--seed', type=int, default=17, metavar='S',
                    help='random seed (default: 17)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data_attack/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='../data_attack/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = lrat_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           training_version=args.training_version,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
                           
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >=90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def cwloss(output, target, confidence=50,num_classes=10):
        # we reused the implementation in the following repo:  CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT

        # compute the probability of the label class versus the maximum other
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss

      
#PGDL_2 attack
def l2_attack(model, X, y, epsilon= 0.5, perturb_steps = 20):
     delta = 0.001 * torch.randn(X.shape).cuda().detach()
     delta = Variable(delta.data, requires_grad=True)
     out = model(X)
     err = (out.data.max(1)[1] != y.data).float().sum()
     batch_size = len(X)
        # Setup optimizers
     optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

     for _ in range(perturb_steps):
         adv = X + delta

            # optimize
         optimizer_delta.zero_grad()
         with torch.enable_grad():
             loss =  (-1)* nn.CrossEntropyLoss()(model(adv), y)

         loss.backward()
            # renorming gradient
         grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
         delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                       # avoid nan or inf if gradient is 0
         if (grad_norms == 0).any():
             delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
         optimizer_delta.step()

            # projection
         delta.data.add_(X)
         delta.data.clamp_(0, 1).sub_(X)
         delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
     x_adv = Variable(X + delta, requires_grad=False)
     x_adv = torch.clamp(x_adv, 0.0, 1.0)
     err_pgd = (model(x_adv).data.max(1)[1] != y.data).float().sum()
     return err, err_pgd


    
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
           loss = nn.CrossEntropyLoss()(model(X_pgd), y)
          # loss = torch.log(nn.CrossEntropyLoss()(model(X_pgd), y))/torch.log(nn.CrossEntropyLoss()(model(X_pgd), y))
          #  loss = cwloss(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


# initialize source model
model_2 = ResNet18().to(device)

def _pgd_blackbox(model, model_2,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_2(X_pgd), y)
           #  loss = cwloss(model50(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd




def _fgsm_whitebox(model, X, y, epsilon = 0.031):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    deltasign =  epsilon * delta.grad.detach().sign()

    out_f = model(X)
    err_f = (out_f.data.max(1)[1] != y.data).float().sum()
    x_adv = torch.clamp(X+deltasign, 0.0, 1.0)
    out_fg = model(x_adv)

    err_fg = (out_fg.data.max(1)[1] != y.data).float().sum()
    return err_f, err_fg





def eval_adv_test_whitebox(model, device, test_loader):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model,X, y)
           
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(test_loader.dataset))
    return 1 - natural_err_total / len(test_loader.dataset), 1- robust_err_total / len(test_loader.dataset)


def main():

    model = ResNet18().to(device)
   
   # model2 = ResNet18().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = args.momentum ,weight_decay=args.weight_decay)   
    
    natural_acc = []
    robust_acc = []
    rob_acc = 0.5400
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        start_time = time.time()

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)


        print('================================================================')

        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader)

        print('using time:', time.time()-start_time)
        
        natural_acc.append(natural_err_total)
        robust_acc.append(robust_err_total)
        print('================================================================')
        if  robust_err_total > rob_acc:
           rob_acc = robust_err_total
           #torch.save(model.state_dict(), '../SNART/snartcifar100-resnet-18-epoch.pt')
           # torch.save(optimizer.state_dict(), '../SNART/s7resnet-165.tar')
        
        file_name = os.path.join(log_dir, 'train_stats.npy')
        np.save(file_name, np.stack((np.array(natural_acc), np.array(robust_acc))))        

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-res-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res-checkpoint_epoch{}.tar'.format(epoch)))
        # eval_auto_white_eval_auto_whitebox(model, X, y,test_loader)

    # l = [x for (x, y) in test_loader]
    # x_test = torch.cat(l, 0)
    # l = [y for (x, y) in test_loader]
    # y_test = torch.cat(l, 0)
    # adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    #adversary.apgd.n_restarts = 1
    #x_adv = adversary.run_standard_evaluation(x_test, y_test)

if __name__ == '__main__':
    main()



