import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,c=0.1,xi=1e-3,decay=0.5,gamma=1000.0,args=None,use_cuda=True):
        self.model=model
        self.model_old=None
        
        self.omega={}
        self.DELTA={}
        self.OMEGA={}
        self.p_old={}

        for n,p in self.model.named_parameters():
            if p.requires_grad:
                self.OMEGA[n] = p.data.clone().zero_()

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad
        self.logger = args.logger

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.c=c
        self.xi=xi
        self.decay=decay
        self.gamma=gamma
        if args:
            if hasattr(args, 'c'):
                self.c=args.c
            if hasattr(args, 'xi'):
                self.xi=args.xi
            if hasattr(args, 'decay'):
                self.decay=args.decay
            print('Setting parameters to c-'+str(self.c)+', xi-'+str(self.xi)+', decay-'+str(self.decay))

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # reset importance omega
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] = p.data.clone().zero_()
                self.DELTA[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            self.train_epoch(t,xtrain,ytrain,e)
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            
            self.logger.log_scalar(str(t)+"_train acc", train_acc, e)
            self.logger.log_scalar(str(t)+"_valid acc", valid_acc, e)
            self.logger.log_scalar(str(t)+"_train loss", train_loss, e)
            self.logger.log_scalar(str(t)+"_valid loss", valid_loss, e)
            
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer=self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        # Update task regularization OMEGA
        for (n,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
            if p.requires_grad:
                #change = param.detach().clone() - param_old
                #o = torch.nn.functional.relu(self.omega[n])/(change.pow(2) + self.xi)
                o = torch.nn.functional.relu(self.omega[n])/(self.DELTA[n].pow(2) + self.xi)
                self.OMEGA[n] = self.OMEGA[n]*self.decay+o*(1-self.decay) #self.OMEGA[n] + o #

        return

    def train_epoch(self,t,x,y,e):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            with torch.no_grad():
                images=torch.autograd.Variable(x[b])
                targets=torch.autograd.Variable(y[b])

            # Forward current model
            outputs= self.model.forward(images, t)
            output=outputs[t]
            loss=self.criterion(t,output,targets,len(r)*e + i)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            
            self.optimizer.step()
            self.logger.inc_iter()

            # track path integral
            for n,p in self.model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        change = (p.detach()-self.p_old[n])
                        self.omega[n] += (-p.grad*change)
                        self.DELTA[n] += -p.grad*change
                self.p_old[n] = p.detach().clone()


        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            with torch.no_grad():
                images=torch.autograd.Variable(x[b])
                targets=torch.autograd.Variable(y[b])

            # Forward
            outputs= self.model.forward(images, t)
            output=outputs[t]
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*len(b)
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets, i=None):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (n,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                if param.requires_grad:
                    loss_reg += torch.sum(self.OMEGA[n].data*(param_old.data-param).pow(2))

            if i and i % 4000 == 0:
                print(loss_reg)

        c1_mean = self.model.c1_xda.reg(t)
        c2_mean = self.model.c2_xda.reg(t)
        c3_mean = self.model.c3_xda.reg(t)
        fc1_mean = self.model.fc1_xda.reg(t)
        fc2_mean = self.model.fc2_xda.reg(t)

        reg_dist = sum([c1_mean,
                        c2_mean,
                        c3_mean,
                        fc1_mean,
                        fc2_mean,
                        5.0])

        if i and i % 4000 == 0:
            print(reg_dist)

        err=self.ce(output,targets)

        if i:
            self.logger.log_scalar(str(t)+"_err", err, i)
            self.logger.log_scalar(str(t)+"_reg", loss_reg, i)
            self.logger.log_scalar(str(t)+"_reg_dist", reg_dist, i)
            self.logger.log_scalar(str(t)+"_c1_mean", c1_mean, i)
            self.logger.log_scalar(str(t)+"_c2_mean", c2_mean, i)
            self.logger.log_scalar(str(t)+"_c3_mean", c3_mean, i)
            self.logger.log_scalar(str(t)+"_fc1_mean", fc1_mean, i)
            self.logger.log_scalar(str(t)+"_fc2_mean", fc2_mean, i)


        return err+self.c*loss_reg+0.05*reg_dist
