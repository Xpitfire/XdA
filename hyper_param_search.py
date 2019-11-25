import numpy as np
import sys, os, time
import torch
import utils
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
import traceback
import importlib
import global_params
import monitor


def define_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

dataloader = None
approach = None
network = None
use_cuda = None

def load_cuda_settings():
    global use_cuda
    if torch.cuda.is_available():
        use_cuda = True
        torch.cuda.manual_seed(args.seed)
    else:
        use_cuda = False
        print('[CUDA unavailable]')

def load_experiment(args):
    global dataloader
    if args.experiment == 'cifar':
        from dataloaders import cifar as dataloader
    elif args.experiment == 'mixture':
        from dataloaders import mixture as dataloader
    elif args.experiment == 'mnist2':
        from dataloaders import mnist2 as dataloader
    elif args.experiment == 'pmnist':
        from dataloaders import pmnist as dataloader
    else:
        dataloader = None
        print('Dataset not supported')

def load_network_and_approach(args, approaches_dir='approaches', networks_dir='networks'):
    global network, approach
    network = importlib.import_module('.'+args.network, package=networks_dir)
    approach = importlib.import_module('.'+args.approach, package=approaches_dir)

def train(args):
    # load settings
    define_seed(args)
    load_cuda_settings()
    load_experiment(args)
    load_network_and_approach(args)
    
    tstart=time.time()

    # Load
    print('Load data...')
    data, taskcla, inputsize = dataloader.get(seed=args.seed)
    print('Input size =', inputsize, '\nTask info =', taskcla)

    # Inits
    print('Inits...')
    net = network.Net(inputsize, taskcla)
    if use_cuda:
        net = net.cuda()

    params = utils.calculate_parameters(net)
    print('Num parameters = %s' % (params))
    if args.print_stats:
        utils.print_model_report(net, params)

    appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args, use_cuda=use_cuda)
    if args.print_stats:
        utils.print_optimizer_config(appr.optimizer)
    print('-' * 100)

    try:
        # Loop tasks
        acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        for t, ncla in taskcla:
            #print('*' * 100)
            #print('Task {:2d} ({:s})'.format(t, data[t]['name']))
            #print('*' * 100)

            if args.approach == 'joint':
                # Get data. We do not put it to GPU
                if t == 0:
                    xtrain = data[t]['train']['x']
                    ytrain = data[t]['train']['y']
                    xvalid = data[t]['valid']['x']
                    yvalid = data[t]['valid']['y']
                    task_t = t * torch.ones(xtrain.size(0)).int()
                    task_v = t * torch.ones(xvalid.size(0)).int()
                    task = [task_t, task_v]
                else:
                    xtrain = torch.cat((xtrain, data[t]['train']['x']))
                    ytrain = torch.cat((ytrain, data[t]['train']['y']))
                    xvalid = torch.cat((xvalid, data[t]['valid']['x']))
                    yvalid = torch.cat((yvalid, data[t]['valid']['y']))
                    task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
                    task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
                    task = [task_t, task_v]
            else:
                # Get data
                xtrain = data[t]['train']['x']
                ytrain = data[t]['train']['y']
                xvalid = data[t]['valid']['x']
                yvalid = data[t]['valid']['y']
                if use_cuda:
                    xtrain = xtrain.cuda()
                    ytrain = ytrain.cuda()
                    xvalid = xvalid.cuda()
                    yvalid = yvalid.cuda()

                task = t

            # Train
            appr.train(task, xtrain, ytrain, xvalid, yvalid)
            #print('-' * 100)

            # Test
            for u in range(t + 1):
                xtest = data[u]['test']['x']
                ytest = data[u]['test']['y']
                if use_cuda:
                    xtest = xtest.cuda()
                    ytest = ytest.cuda()
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                #print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
                acc[t, u] = test_acc
                lss[t, u] = test_loss

            # Save
            print('Save at ' + args.output)
            np.savetxt(args.output, acc, '%.4f')
    except:
        traceback.print_exc()
        return net

    # Done
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    if hasattr(appr, 'logs'):
        if appr.logs is not None:
            # save task names
            from copy import deepcopy
            appr.logs['task_name'] = {}
            appr.logs['test_acc'] = {}
            appr.logs['test_loss'] = {}
            for t, ncla in taskcla:
                appr.logs['task_name'][t] = deepcopy(data[t]['name'])
                appr.logs['test_acc'][t] = deepcopy(acc[t, :])
                appr.logs['test_loss'][t] = deepcopy(lss[t, :])
            # pickle
            import gzip
            import pickle
            with gzip.open(os.path.join(appr.logpath), 'wb') as output:
                pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

    return net

args = global_params.args
args.seed = 0
args.experiment = 'cifar'
args.approach = 'si'
args.network = 'alexnet_' + args.approach
args.print_stats = False
args.nepochs = 5
args.lr = 0.05
args.enable_logging = True
args.log_steps = 1000
args.log_dir = 'hyper_param_runs'

Xi = [0.001, 0.01, 0.1]
C = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]

for xi in Xi:
    for c in C:
        args.xi = xi
        args.c = c
        args.comment = args.approach+'_xi-'+str(xi)+'_c-'+str(c)
        args.output = 'hyper_param_runs/res/'+args.network+'_'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'_xi-'+str(xi)+'_c-'+str(c)+'.txt'
        args.logger = monitor.Logger(args)
        net = train(args)