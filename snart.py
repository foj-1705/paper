import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np




def lrat_loss(model,
              x_natural,
              y,
              optimizer,
              training_version,
              beta = 0.4,
              step_size=0.003,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf'):
    kl = nn.KLDivLoss(size_average=False)
    model.eval()
    
    criterion_kl = nn.KLDivLoss(size_average = False)
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                                       
                loss_ce =  F.cross_entropy(model(x_adv), y)
            
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
   
   

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    

    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)
    #logits_new = model(x_new)
    
      
    logit_diff = logits_adv - logits

    huber = nn.SmoothL1Loss()

    mse_loss = nn.MSELoss()   

    adv_loss = F.cross_entropy(logits_adv, y)   
    nat_loss = F.cross_entropy(logits, y)
    
   
    adv_prob = F.softmax(logits_adv, dim=1)
    adv_prob1 = torch.max(adv_prob) 
    prob = F.softmax(logits, dim=1)
    prob1 = torch.max(prob)


    
   
    if training_version == 'lrat':
       loss = nat_loss + beta*(adv_loss/nat_loss)
    elif training_version == 'lrllat': 
       loss = nat_loss + beta*(adv_loss/nat_loss)+ 1.0*mse_loss(logits_adv, logits)
    else:
       loss = adv_loss  #defaults to standard AT
     
 
   
    return loss











