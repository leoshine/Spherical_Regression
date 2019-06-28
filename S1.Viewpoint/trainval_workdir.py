"""
 @Author  : Shuai Liao
"""

import torch
import torch.optim
from torch.utils.data.sampler import RandomSampler, BatchSampler

from basic.common import Open, env, add_path, RefObj as rdict, argv2dict
from basic.util   import load_yaml, parse_yaml
import os, sys, time, shutil
from time import gmtime, strftime

import numpy as np
import pickle
from math import pi
from easydict import EasyDict as edict
from ordered_easydict import Ordered_EasyDict as oedict
from pprint import pprint

from tensorboardX import SummaryWriter
from tqdm import tqdm

this_dir = os.path.dirname(os.path.realpath(__file__))
add_path(this_dir+'/../dataset')
from Pascal3D import categories


this_dir = os.path.dirname(os.path.abspath(__file__))
add_path(this_dir)
from pytorch_util.libtrain.tools import get_stripped_DataParallel_state_dict, patch_saved_DataParallel_state_dict

from lib.eval.eval_aet_multilevel import eval_cates, compute_geo_dists
from txt_table_v1 import TxtTable


DEBUG = True # False


#------- args from convenient run yaml ---------
# For the purpose that no need to specific each run (without argparse)
convenience_run_argv_yaml=\
'''
net_arch: {net_arch}

net_module: reg_Euler2D_Sexp_Net

#----------------------------------------------
# use gpu
use_gpu    : True
gpu_ids    : [ ] # $mGPUs

# pytorch specifics
num_workers: 3
pin_memory : True

#---[solver control]
niter     : 40000  # 50000
snapshot  : 20000

test_step : 200

base_lr   : 0.001
momentum  : 0.9
weight_decay: 0.0005  # 0.00001

with_loss : Yes
with_pred : No
with_acc  : No
#-----------------------------------------------

RNG_SEED  : 3

cates     :   # Leave empty will return None. By default it will use all 12 categories.

out_rslt_path : $this_dir/$work_dir.rslt.cache.txt
out_eval_path : $this_dir/$work_dir.eval.cache.txt


pretrained : torchmodel   # caffemodel

with_syn   : True
with_flip  : False

'''.format( net_arch='resnet101',  #'ResNet18',
           )
run_args = parse_yaml(convenience_run_argv_yaml)  # odict


# optionally resume from a checkpoint
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--work_dir', default='./snapshots', type=str, metavar='PATH',)
parser.add_argument('--resume'  , action="store_true", default=False,
                    help='to resume by the last checkpoint')
parser.add_argument('--pretrain', default=None, type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
_pargs, _rest = parser.parse_known_args() # parser.parse_args()
# parse the rest undefined args with "--key=value" form.
_cmd_args = argv2dict(_rest)  # odict
_cmd_args.update( vars(_pargs) )


run_args.update(_cmd_args)
opt = oedict(run_args) # easydict updata bug. just re-instantiate it.


if opt.cates is None:
    opt.cates = categories
cates = opt.cates

#------ Import modules ----------
from lib.datasets import dataset_naiReg

_dataset_module = dataset_naiReg.Dataset_regSquaredProbV2
_cfg            = dataset_naiReg.netcfg[opt.net_arch]

np.random.seed(opt.RNG_SEED)
torch.manual_seed(opt.RNG_SEED)
if opt.use_gpu:
    torch.cuda.manual_seed(opt.RNG_SEED)

#---------------------------------------------------------------------------------------------------[dataset]
dataset_kwargs =dict(net_arch=opt.net_arch, with_aug=False, with_flip=opt.with_flip,  mode=opt.pretrained,
                     sampling=dict(pascalvoc=1.0, imagenet=1.0, synthetic=1.0 if opt.with_syn else 0.0) )
pprint(dataset_kwargs)
dataset_train = _dataset_module( 'train', cates=opt.cates, **dataset_kwargs) #
dataset_test  = _dataset_module( 'val',   cates=opt.cates, net_arch=opt.net_arch, with_flip=False, mode=opt.pretrained)

opt.work_dir  += '/Pascal3D_' + ('with_syn' if opt.with_syn else 'only')       # sys.argv[2]
if opt.with_syn:
    opt.niter *= 2
    print('Set niter=%s' % opt.niter)

nr_GPUs = len(opt.gpu_ids)
# assert nr_GPUs>=1, opt.gpu_ids
if nr_GPUs>1:
    print ('---------------------   Use multiple-GPU %s   -------------------------' % opt.gpu_ids)
    print ('     batch_size  = %s' % (_cfg.TRAIN.BATCH_SIZE*nr_GPUs))
    print ('     num_workers = %s' % (opt.num_workers*nr_GPUs)      )

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=_cfg.TRAIN.BATCH_SIZE*max(1,nr_GPUs), shuffle=True, drop_last=True,
                                           num_workers=opt.num_workers*max(1,nr_GPUs), pin_memory=opt.pin_memory, sampler=None)
test_loader  = torch.utils.data.DataLoader(dataset_test,  batch_size=_cfg.TEST.BATCH_SIZE*max(1,nr_GPUs),  shuffle=False, drop_last=False,
                                           num_workers=opt.num_workers*max(1,nr_GPUs), pin_memory=opt.pin_memory, sampler=None)


#---------------------------------------------------------------------------------------------------[model]
add_path(os.path.join(this_dir, 'models'))
import regEulerNet

_net_module = regEulerNet.__getattribute__(opt.net_module)
print ('[makenet] nr_cate: ', len(cates))
model = _net_module(net_arch=opt.net_arch, nr_cate=len(cates), pretrained=opt.pretrained)
#
watch_targets = model.targets
opt.work_dir  += '/%s/%s.%s' % (opt.net_module, opt.net_arch, opt.pretrained)

#---------------------------------------------------------------------------------------------------[optimizer]
params = []
for name, param in model.named_parameters():
    # print ('----(*)  ', name)
    if param.requires_grad:
        params.append(param)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD( params, opt.base_lr,
                             momentum=opt.momentum,
                             weight_decay=opt.weight_decay )

# global state variables.
start_it    = 0 # -1
start_epoch = 0 # -1
from pytorch_util.libtrain import rm_models, list_models

# Log file.
script_name, _ = os.path.splitext( os.path.basename(__file__) )
log_filename   = '%s/%s.log' % (opt.work_dir,script_name)
if os.path.exists(log_filename): # backup previous content.
    pre_log_content = open(log_filename).read()
logf = Open(log_filename, 'w')
def logprint(s):
    print ("\r%s                            " % s)
    logf.write(s+"\n")

#-- Resume or use pretrained (Note not imagenet pretrain.)
assert not (opt.resume and opt.pretrain is not None), 'Only resume or pretrain can exist.'
if opt.resume:
    iter_nums, net_name = list_models(opt.work_dir) # ('snapshots')
    assert len(iter_nums)>0, "No models available"
    latest_model_name = os.path.join(opt.work_dir, '%s_iter_%s.pth.tar' % (net_name, iter_nums[-1]))

    print ('\n\nResuming from: %s \n\n' % latest_model_name)
    if os.path.isfile(latest_model_name):
        print("=> loading checkpoint '{}'".format(latest_model_name))
        checkpoint = torch.load(latest_model_name)
        start_it, start_epoch = checkpoint['it_and_epoch']  # mainly for control lr: (it, epoch)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (it_and_epoch {})"
              .format(latest_model_name, checkpoint['it_and_epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(latest_model_name))  # unnecessary line
elif opt.pretrain is not None:
    print ('\n\nUsing pretrained: %s \n\n' % opt.pretrain)
    if os.path.isfile(opt.pretrain):
        print("=> loading checkpoint '{}'".format(opt.pretrain))
        checkpoint = torch.load(opt.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (it_and_epoch {})"
              .format(opt.pretrain, checkpoint['it_and_epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.pretrain))



# auto cpu, gpu mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Check if use multi-gpus.
# Note should be after any "model.load_state_dict()" call!
if opt.use_gpu:
    if nr_GPUs>1: # multi-GPUs   opt['mGPUs']:
        # see: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        if   opt.net_arch.startswith('alexnet'):
            model.trunk.Convs = torch.nn.DataParallel(model.trunk.Convs)
        elif opt.net_arch.startswith('vgg'):
            model.trunk.features = torch.nn.DataParallel(model.trunk.features)
        else:
            model.trunk = torch.nn.DataParallel(model.trunk)
    model.to(device)  #cuda()


if not os.path.exists(opt.work_dir):
    print ("[Make new dir] ", opt.work_dir)
    os.makedirs(opt.work_dir)



disp_interval = 10  if ('disp_interval' not in opt)  else opt['disp_interval']


def adjust_learning_rate_by_iter(optimizer, cur_iter, max_iter):
    """Sets the learning rate to the initial LR decayed by 10 every _N_ epochs"""
    _N_ = max_iter // 3  # add just learning rate 3 times.
    lr = opt.base_lr * (0.1 ** (max(cur_iter,0) // _N_)) # 300))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(niter, disp_interval=40):

    os.system('rm -rf %s ' % (opt.work_dir+"/logs") )
    logger = SummaryWriter(opt.work_dir+"/logs")

    pre_time = time.time()
    to_continue = True
    it    = start_it    -1 # -1
    epoch = start_epoch -1 # -1
    lr    = adjust_learning_rate_by_iter(optimizer, it, niter)  # opt.base_lr
    while to_continue:
        epoch += 1

        # for _i_, sample_batched in enumerate(train_loader):
        pbar = tqdm(train_loader)
        for _i_, sample_batched in enumerate(pbar):
            pbar.set_description("[work_dir] %s   B=%s " % (os.path.relpath(opt.work_dir), _cfg.TRAIN.BATCH_SIZE))
            rec_inds = sample_batched['idx'].numpy()

            # switch to train mode
            model.train()

            it += 1
            if it % 1000==0:
                lr = adjust_learning_rate_by_iter(optimizer, it, niter)

            to_continue = it<niter
            if not to_continue:
                save_checkpoint({
                    'it_and_epoch': (it, epoch),
                    'state_dict': model.state_dict(),  # get_stripped_DataParallel_state_dict(model), # model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, filename=os.path.join(opt.work_dir,'train_iter_%s.pth.tar'%(it)))  # (it+1)
                # break for/while loop.
                break  # Note: this may cause an exception of tqdm process at the end training. We ignore it for now.

            data  = sample_batched['data' ].to(device, non_blocking=True)
            label = sample_batched['label'].to(device, non_blocking=True)
            # formulate GT dict
            _gt_targets = model.gt_targets if hasattr(model, 'gt_targets') else model.targets
            GT    = edict()
            for tgt in _gt_targets:
                GT[tgt] = sample_batched[tgt].to(device, non_blocking=True)

            # compute Pred output
            Prob = model(data, label)

            # compute Loss for each target and formulate Loss dictionary.
            Loss = model.compute_loss(Prob, GT)

            total_loss = 0
            for tgt in watch_targets:
                total_loss += Loss[tgt]

            # compute gradient and do SGD step
            optimizer.zero_grad()  # Clears the gradients.
            total_loss.backward()

            optimizer.step()

            if not DEBUG:
                logger.add_scalars('loss_iter', Loss, it+1)

            # print loss info
            if (it+1) % disp_interval == 0 or (it+1) == niter:
                cur_time = time.time()
                time_consume = cur_time - pre_time
                pre_time = cur_time
                logprint( '%s [epoch] %3d [iter] %5d / %d -------------------[time_consume] %.2f   lr=%.e'
                     % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), epoch, it, opt.niter, time_consume, lr) )
                for tgt in watch_targets:
                    _loss = Loss[tgt].data.cpu().numpy().copy()
                    logprint( '  %-15s  loss=%.3f' % (tgt, _loss) )
                    if np.isnan(_loss):
                        print ("[Warning]  Weights explode!  Stop training ... ")
                        exit(-1)

                # Compute Acc@theta
                recs = dataset_train.recs[rec_inds]
                Pred = model.compute_pred(Prob)
                # convert prediction back to [0, 360]
                Pred = dataset_test.pred2angle(Pred['a'].cpu().data.numpy(),
                                               Pred['e'].cpu().data.numpy(),
                                               Pred['t'].cpu().data.numpy())
                GT   = recs.gt_view['a'], recs.gt_view['e'], recs.gt_view['t']
                geo_dists = compute_geo_dists(GT, Pred)
                MedError  = np.median(geo_dists) /np.pi*180.
                theta_levels = dict(zip(['pi/6','pi/12','pi/24'],[np.pi/6,np.pi/12,np.pi/24]))
                Acc_at_ts = dict([(tname,sum(geo_dists<tvalue)/float(len(geo_dists))) for tname,tvalue in theta_levels.items()])
                logger.add_scalars('acc/train', Acc_at_ts, it+1)
                acc_str = '   '.join(['[%s] %3.1f%%' %(k,Acc_at_ts[k]*100) for k,v in theta_levels.items()])
                logprint('  Acc@{ %s }   '%acc_str)
                sys.stdout.flush()
                #


            if (it+1)%opt.snapshot==0:
                # print '\n\n\nSaving model ....\n\n\n'
                save_checkpoint({
                    'it_and_epoch': (it, epoch),
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, filename=os.path.join(opt.work_dir,'train_iter_%s.pth.tar'%(it+1)))


            # Do test
            if it % opt.test_step==0:
                _MedError, _Acc_pi_6, _Acc_pi_12, _Acc_pi_24 = test()
                logger.add_scalars('acc/test', {'pi/6':_Acc_pi_6, 'pi/12':_Acc_pi_12, 'pi/24':_Acc_pi_24}, it+1)



def test():

    out_rslt_path = os.path.join(opt.work_dir, 'rslt.cache.txt')
    out_eval_path = os.path.join(opt.work_dir, 'eval.cache.txt')
    keys = dataset_test.keys

    # switch to train mode
    model.eval()

    # # loss reducer
    from pytorch_util.libtrain.reducer import reducer, reducer_group
    gLoss_redu = reducer_group(*watch_targets)
    gPred_redu = reducer_group(*['a','e','t'])

    pre_time = time.time()
    it = -1
    epoch = -1

    # for _i_, sample_batched in enumerate(test_loader):

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for _i_, sample_batched in enumerate(pbar):
            pbar.set_description("[work_dir] %s  " % os.path.relpath(opt.work_dir))

            it += 1

            # prepare input data
            label = sample_batched['label'].to(device, non_blocking=True)
            data  = sample_batched['data' ].to(device, non_blocking=True)
            # formulate GT dict
            _gt_targets = model.gt_targets if hasattr(model, 'gt_targets') else model.targets
            GT    = edict()
            for tgt in _gt_targets:
                GT[tgt] = sample_batched[tgt].to(device, non_blocking=True)

            # compute Pred output
            Prob = model(data, label)

            # compute Loss for each target and formulate Loss dictionary.
            Loss = model.compute_loss(Prob, GT)

            total_loss = 0
            for tgt in watch_targets:
                total_loss += Loss[tgt]  # * loss_weight

            # predict target angles value
            Pred = model.compute_pred(Prob)

            gLoss_redu.collect(Loss) # pass in dict of all loss (loss_a, loss_e, loss_t).
            gPred_redu.collect(Pred)

            # print loss info
            cur_time = time.time()
            time_consume = cur_time - pre_time
            pre_time = cur_time


    name2pred = gPred_redu.reduce()

    # convert prediction back to [0, 360]
    a, e, t = dataset_test.pred2angle(name2pred['a'], name2pred['e'], name2pred['t'])  # or call _dataset_module.pred2angle

    #-- Write result to file  (Format: # {obj_id}  {a} {e} {t} )
    rslt_lines  = '%-40s  %5s  %5s  %5s\n' % ('# obj_id','a','e','t')
    rslt_lines += ''.join(['%-40s  %5.1f  %5.1f  %5.1f\n' % (_k,_a,_e,_t) for _k,_a,_e,_t in zip(keys, a, e, t)])
    Open(out_rslt_path,'w').write(rslt_lines)
    print ("[output] ", out_rslt_path)

    #-- Do evaluation  ('MedError', 'Acc@theta')
    summary_str = eval_cates(out_rslt_path, cates=opt.cates, theta_levels_str='pi/6  pi/12  pi/24') # ['aeroplane','boat','car'])
    Open(out_eval_path,'w').write(summary_str)
    print(summary_str)

    reca = TxtTable().load_as_recarr(out_eval_path, fields=['MedError', 'Acc@pi/6', 'Acc@pi/12', 'Acc@pi/24'])

    return reca[-1]


if __name__ == '__main__':
    train(niter=opt.niter)
    test()
