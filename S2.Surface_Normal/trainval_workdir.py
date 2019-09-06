"""
 @Author  : Shuai Liao
"""

import matplotlib
matplotlib.use('Agg')  # server mode.
#
import torch
import torch.optim
from torch.utils.data.sampler import RandomSampler, BatchSampler
#
import re
import os, sys, time, shutil
from time import gmtime, strftime
import numpy as np
from math import pi
from easydict import EasyDict as edict
from collections import OrderedDict as odict
#
from basic.common import Open, env, add_path, RefObj as rdict, argv2dict
#
from pytorch_util.libtrain.yaml_netconf import parse_yaml, import_module_v2
from pytorch_util.libtrain.tools import get_stripped_DataParallel_state_dict, patch_saved_DataParallel_state_dict
from txt_table_v1 import TxtTable
#
from tensorboardX import SummaryWriter


#===========  Parsing from working path =========
base_dir = os.path.dirname(os.path.abspath(__file__))
# parsing current dir
pwd      = os.getcwd()
MtdFamily, MtdType  = pwd.split(os.sep)[-2:]
#================================================
add_path(base_dir+'/lib')


#------- args from convenient run yaml ---------
# For the purpose that no need to specific each run (without argparse)
convenience_run_argv_yaml=\
'''
MtdFamily       : {MtdFamily}
MtdType         : {MtdType}
net_module      : {net_module}

net_arch        : {net_arch}
base_dir        : {base_dir}
LIB_DIR         : {base_dir}/lib

work_dir        : './snapshots/{net_arch}'
nr_epoch        : 20 # 60
test_step_epoch : 1  # 10

nr_disp         : 500
disp_interval   : 100  # every 100 iters disp.  Warning: Overwrite nr_disp.

nr_adjust_lr    : 3

train_with_flip : True
work_dir_suffix : ''

sample_nyu      : 1.0
sample_syn      : 0.0

Ladicky_normal  : True

this_dir        : {this_dir}
base_lr         : 0.0001
'''.format(net_arch='vgg16',
           base_dir=base_dir,
           this_dir=pwd,
           #
           net_module=MtdFamily,
           MtdFamily =MtdFamily,
           MtdType   =MtdType,
           )
run_args = parse_yaml(convenience_run_argv_yaml)  # odict


#------- arg from argparse -----------------
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('work_dir'     , default='', type=str, metavar='PATH',)
parser.add_argument('device_ids'   , default='', type=str, metavar='PATH',
                    help='e.g. 0  or 0,1,2,3')
parser.add_argument('--resume'     , action="store_true", default=False,
                    help='to resume by the last checkpoint')
parser.add_argument('--pretrain'   , default=None, type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--optimizer', default='Adam', type=str, help='SGD or Adam')
parser.add_argument('--test_only'   , action="store_true", default=False,
                    help='only do test once.')
parser.add_argument('--test_save_pred'   , action="store_true", default=False,
                    help='save prediction as lmdb.')
_pargs, _rest = parser.parse_known_args()
_cmd_args = argv2dict(_rest)  # odict
_cmd_args.update( vars(_pargs) )
#
run_args.update(_cmd_args)

from string import Template
template_str = open( os.path.join(base_dir, 'conf_template.yml')).read()
template = Template(template_str)
conf_yml_str = template.substitute(run_args)

#-- parse module_yml_file
opt = parse_yaml(conf_yml_str )
#
opt.update( run_args )
#
from ordered_easydict import Ordered_EasyDict as oedict
opt = oedict(opt) # Use opt for reference all configurations.


#------ Import modules ----------
[(_dataset_module,_dataset_kwargs), netcfg] = import_module_v2(opt.IMPORT_dataset)
[(_net_module,_net_kwargs) ] = import_module_v2(opt.IMPORT_makenet)
[(eval_all,_), ]             = import_module_v2(opt.IMPORT_eval.lmdb)
net_arch = opt.net_arch

_cfg = netcfg[net_arch]
np.random.seed(_cfg.RNG_SEED)
torch.manual_seed(_cfg.RNG_SEED)
if opt.use_gpu:
    torch.cuda.manual_seed(_cfg.RNG_SEED)

#---------------------------------------------------------------------------------------------------[dataset]
dataset_test  = _dataset_module( collection='test' , Ladicky_normal=opt.Ladicky_normal, sampling=dict(nyu=1.0, syn=0.0), **_dataset_kwargs) #
dataset_train = _dataset_module( collection='train', Ladicky_normal=opt.Ladicky_normal, sampling=dict(nyu=opt.sample_nyu, syn=opt.sample_syn),
                                 with_flip=opt.train_with_flip, **_dataset_kwargs)
# 'ModelNet10/SO3_100V.white_BG_golden_FG'

if 'batch_size' in opt:
    batch_size = opt.batch_size
else:
    batch_size = _cfg.TRAIN.BATCH_SIZE
nr_GPUs = len(opt.gpu_ids)
assert nr_GPUs>=1, opt.gpu_ids
if nr_GPUs>1:
    print ('---------------------   Use multiple-GPU %s   -------------------------' % opt.gpu_ids)
    print ('     batch_size  = %s' % (batch_size*nr_GPUs))
    print ('     num_workers = %s' % (opt.num_workers*nr_GPUs))
#
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size*nr_GPUs, shuffle=True,
                                           num_workers=opt.num_workers*nr_GPUs, pin_memory=opt.pin_memory, sampler=None)


# auto cpu, gpu mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------------------------------------------------------------------------------------------[model]
print ('[makenet] ')
model = _net_module(**_net_kwargs)
if 'fix_conv1_conv2' in opt.keys() and opt.fix_conv1_conv2:
    model.fix_conv1_conv2()
#
watch_targets = model.targets

#---------------------------------------------------------------------------------------------------[optimizer]
params = []
for name, param in model.named_parameters():
    # print '----(*)  ', name
    if param.requires_grad:
        params.append(param)

print('[Optimizer] %s' % opt.optimizer)
if   opt.optimizer=='Adam':
    optimizer = torch.optim.Adam(params, lr=opt.base_lr)
elif opt.optimizer=='SGD':
    optimizer = torch.optim.SGD( params, opt.base_lr,
                                 momentum=opt.momentum,
                                 weight_decay=opt.weight_decay )
else:
    raise NotImplementedError


work_dir = opt.work_dir +'.N%.2f_S%.2f' % (opt.sample_nyu, opt.sample_syn)
if opt.work_dir_suffix != '':
    work_dir += '.%s' % opt.work_dir_suffix


# global state variables.
start_it    = 0
start_epoch = 0
from pytorch_util.libtrain import rm_models, list_models
from pytorch_util.libtrain.reducer import reducer, reducer_group

# Log file.
script_name, _ = os.path.splitext( os.path.basename(__file__) )
log_filename   = '%s/%s.log' % (work_dir,script_name)
if os.path.exists(log_filename): # backup previous content.
    pre_log_content = open(log_filename).read()
logf = Open(log_filename, 'w')
def logprint(s):
    print ("\r%s                            "%s)
    logf.write(s+"\n")

#-- Resume or use pretrained (Note not imagenet pretrain.)
assert not (opt.resume and opt.pretrain is not None), 'Only resume or pretrain can exist.'
if opt.resume:
    iter_nums, net_name = list_models(work_dir) # ('snapshots')
    assert len(iter_nums)>0, "No models available"
    latest_model_name = os.path.join(work_dir, '%s_iter_%s.pth.tar' % (net_name, iter_nums[-1]))

    print ('\n\nResuming from: %s \n\n' % latest_model_name)
    if os.path.isfile(latest_model_name):
        print("=> loading checkpoint '{}'".format(latest_model_name))
        checkpoint = torch.load(latest_model_name)
        opt.nr_adjust_lr = checkpoint['nr_adjust_lr']
        start_it, start_epoch = checkpoint['nr_it_and_epoch']  # mainly for control lr: (it, epoch)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # fix for pytorch 4.0.x  [https://github.com/jwyang/faster-rcnn.pytorch/issues/222]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print("=> loaded checkpoint '{}' (nr_it_and_epoch {})"
              .format(latest_model_name, checkpoint['nr_it_and_epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(latest_model_name))  # unnecessary line
elif opt.pretrain is not None:
    print ('\n\nUsing pretrained: %s \n\n' % opt.pretrain)
    if os.path.isfile(opt.pretrain):
        print("=> loading checkpoint '{}'".format(opt.pretrain))
        checkpoint = torch.load(opt.pretrain)
        #print checkpoint['state_dict'].keys()
        #exit()
        model.load_state_dict(patch_saved_DataParallel_state_dict(checkpoint['state_dict']))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (nr_it_and_epoch {})"
              .format(opt.pretrain, checkpoint['nr_it_and_epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.pretrain))



# Check if use multi-gpus.
# Note should be after any "model.load_state_dict()" call!
if opt.use_gpu:
    if nr_GPUs>1: # multi-GPUs   opt['mGPUs']:
        model.trunk = torch.nn.DataParallel(model.trunk)
    model.to(device)  # .cuda()


if not os.path.exists(work_dir):
    print( "[Make new dir] ", work_dir)
    os.makedirs(work_dir)



def adjust_learning_rate_by_epoch(optimizer, cur_epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every _N_ epochs"""
    _N_ = max_epoch // opt.nr_adjust_lr  # add just learning rate 3 times.
    lr = opt.base_lr * (0.1 ** (max(cur_epoch,0) // _N_)) # 300))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def test(dataset_test, work_dir, test_model=None, marker='epoch', save_pred=False, train_epoch=None):
    out_rslt_path = work_dir+'/temp.out_rslt_path.txt'
    out_eval_path = work_dir+'/temp.out_eval_path.txt'

    if test_model is None:
        test_model = model
        #---- Load trained weights here.------
        assert os.path.exists(work_dir)
        #
        iter_nums, net_name = list_models(work_dir, marker=marker)
        saved_iter_num = iter_nums[-1]
        pretrained_model = work_dir+'/%s_%s_%s.pth.tar' % (net_name, marker, saved_iter_num) # select maxmun iter number.
        print ('[pretrained_model] ', pretrained_model)

        checkpoint = torch.load(pretrained_model) # load weights here.
        _state_dict = patch_saved_DataParallel_state_dict(checkpoint['state_dict'])
        test_model.load_state_dict(_state_dict)

    # switch to train mode
    test_model.eval()
    gEval_redu = reducer_group(*['Es'])

    pre_time = time.time()
    it = -1
    epoch = -1
    #
    keys = dataset_test.keys
    test_loader = torch.utils.data.DataLoader( dataset_test, batch_size=_cfg.TEST.BATCH_SIZE*nr_GPUs, shuffle=False,
                                               num_workers=opt.num_workers*nr_GPUs, pin_memory=opt.pin_memory, sampler=None)

    if save_pred:
        from lmdb_util import NpyData_lmdb, ImageData_lmdb
        pred_db_path = os.path.join(work_dir, 'PredNormal.Rawpng.lmdb')
        pred_db = ImageData_lmdb(pred_db_path, 'w')

    for _i_, sample_batched in enumerate(test_loader):
        #
        rec_inds = sample_batched['idx'].numpy()
        #
        it += 1

        # Note: Tensor.cuda()   Returns a copy of this object in CUDA memory.
        data = sample_batched['data' ].to(device, non_blocking=True)
        # formulate GT dict
        _gt_targets = model.gt_targets if hasattr(model, 'gt_targets') else model.targets
        GT    = edict()
        GT['mask'] = sample_batched['mask'].to(device, non_blocking=True)
        for tgt in _gt_targets:
            GT[tgt] = sample_batched[tgt].to(device, non_blocking=True)

        # compute Pred output
        Prob = test_model(data)

        # compute Loss for each target and formulate Loss dictionary.
        Loss, _Metric_ = test_model.compute_loss(Prob, GT)

        total_loss = 0
        for tgt in watch_targets:
            total_loss += Loss[tgt]  # * loss_weight
        if 'norm' in _Metric_.keys():
            _Es = _Metric_['norm'].data.cpu().numpy().copy()
            gEval_redu.collect(dict(Es=_Es), squeeze=False)

        # predict as images
        if save_pred:
            Pred = test_model.compute_pred(Prob, encode_bit=8)  #
            predNormImgs = Pred.norm  #  NxHxWx3
            assert len(rec_inds)==len(predNormImgs)
            for i,idx in enumerate(rec_inds):
                key = keys[idx]
                pred_db[key] = predNormImgs[i]

        # print loss info
        cur_time = time.time()
        time_consume = cur_time - pre_time
        pre_time = cur_time
        print( '\r %s [test-iter] %5d / %5d ---------[time_consume] %.2f' % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), it, len(test_loader), time_consume) )
        for trgt in watch_targets:
            _loss = Loss[trgt].data.cpu().numpy().copy()
            print( '  %-15s  loss=%.3f' % (trgt, _loss) )
            if np.isnan(_loss):
                print ("[Warning]  Weights explode!  Stop training ... ")
                exit(-1)

        _watch_targets = watch_targets+['norm'] if 'norm' not in watch_targets else watch_targets
        for tgt in _watch_targets:
            if tgt =='sgc_norm':   # in metric.
                logprint( '   %-10s  [Acc] : %5.1f%%' % (tgt, _Metric_['sgc_norm_acc']*100) )
            else:
                if tgt in _Metric_.keys():
                    Es = _Metric_[tgt].data.cpu().numpy().copy()
                    mean  = np.mean(Es)
                    median= np.median(Es)
                    rmse  = np.sqrt(np.mean(np.power(Es,2)))
                    acc11 = np.mean(Es < 11.25) * 100
                    acc22 = np.mean(Es < 22.5 ) * 100
                    acc30 = np.mean(Es < 30   ) * 100
                    logprint( '   %-10s  [mean]: %5.1f   [median]: %5.1f   [rmse]: %5.1f   [acc{11,22,30}]: %5.1f%%, %5.1f%%, %5.1f%%' % (tgt, mean, median, rmse,acc11,acc22,acc30) )

        print ("\r[work_dir] %s \r" % os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:], end='', flush=True)

    Es = gEval_redu.reduce()['Es']
    mean  = np.mean(Es)
    median= np.median(Es)
    rmse  = np.sqrt(np.mean(np.power(Es,2)))
    acc11 = np.mean(Es < 11.25) * 100
    acc22 = np.mean(Es < 22.5 ) * 100
    acc30 = np.mean(Es < 30   ) * 100
    summary_str = ''
    if train_epoch is not None:
        summary_str += '[Test at epoch %d]\n' % train_epoch
    summary_str += '  [mean]: %5.1f   [median]: %5.1f   [rmse]: %5.1f   [acc{11,22,30}]: %5.1f%%, %5.1f%%, %5.1f%%\n' % (mean, median, rmse,acc11,acc22,acc30)
    logprint( summary_str )
    Open(out_eval_path,'a+').write(summary_str)

    if save_pred:
        mean, median, rmse, acc11, acc22, acc30 = eval_all(pred_db_path, use_multiprocess=False)
        summary_str =  '\n--------------------------------------------'
        summary_str += '[Test] %s \n' % pretrained_model
        summary_str += '  [mean]: %5.1f   [median]: %5.1f   [rmse]: %5.1f   [acc{11,22,30}]: %5.1f%%, %5.1f%%, %5.1f%%\n' % (mean, median, rmse,acc11,acc22,acc30)
        Open(out_eval_path,'a+').write(summary_str)
        print (summary_str)

    return mean, median, rmse,acc11,acc22,acc30

def train():
    # switch to train mode
    # model.train()
    os.system('rm -rf %s ' % (work_dir+"/logs") )
    logger = SummaryWriter(work_dir+"/logs")

    nr_epoch = opt.nr_epoch
    nr_iter  = opt.nr_epoch * (len(dataset_train)/batch_size)  # 130800

    # based on iter
    if 'disp_interval' not in opt:
        disp_interval=nr_iter/opt.nr_disp # calculate based on nr_disp (nr_disp=500)
    else:
        disp_interval=opt.disp_interval   # overwrite by opt setting.
    print ('Start training:    nr_epoch: %d     nr_iter: %d      (disp_interval=%d)' % (nr_epoch, nr_iter, disp_interval))

    pre_time = time.time()
    it    = start_it    # -1
    epoch = start_epoch # -1
    #
    while epoch<nr_epoch:
        # Note here: epoch, it are both 0-based.
        _is_last_epoch = (epoch+1)==nr_epoch

        # switch to train mode
        model.train()
        lr = adjust_learning_rate_by_epoch(optimizer, epoch, nr_epoch)  # opt.base_lr

        for _i_, sample_batched in enumerate(train_loader):
            #ch.now('loader got data.')
            rec_inds = sample_batched['idx'].numpy()
            #
            it += 1
            data = sample_batched['data'].to(device, non_blocking=True)
            # formulate GT dict
            _gt_targets = model.gt_targets if hasattr(model, 'gt_targets') else model.targets
            GT    = edict()
            GT['mask'] = sample_batched['mask'].to(device, non_blocking=True)
            for tgt in _gt_targets:
                GT[tgt] = sample_batched[tgt].to(device, non_blocking=True)

            # compute Pred output
            Prob = model(data)

            # compute Loss for each target and formulate Loss dictionary.
            Loss, _Metric_ = model.compute_loss(Prob, GT)

            total_loss = 0
            for tgt in watch_targets:
                total_loss += Loss[tgt]  # * loss_weight

            # compute gradient and do SGD step
            optimizer.zero_grad()  # Clears the gradients of all optimized Variable s.
            total_loss.backward()

            optimizer.step()

            logger.add_scalars('loss_iter', Loss, it+1)

            # print loss info
            if it % disp_interval == 0: # or (it+1)==len(dataset_train)/batch_size:
                cur_time = time.time()
                time_consume = cur_time - pre_time
                pre_time = cur_time
                logprint( '%s [epoch] %3d/%3d [iter] %5d -----------------------------------[time_consume] %.2f   lr=%.8f'
                     % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), epoch+1, nr_epoch, it+1, time_consume, lr) )
                for tgt in watch_targets:
                    _loss = Loss[tgt].data.cpu().numpy().copy() # Loss['loss_%s'%trgt].data.cpu().numpy().copy()
                    logprint( '  %-15s  loss=%.3f' % (tgt, _loss) )
                    if np.isnan(_loss):
                        print ("[Warning]  Weights explode!  Stop training ... ")
                        exit(-1)

                _watch_targets = watch_targets+['norm'] if 'norm' not in watch_targets else watch_targets
                for tgt in _watch_targets:
                    if tgt =='sgc_norm':   # in metric.
                        logprint( '   %-10s  [Acc] : %5.1f%%' % (tgt, _Metric_['sgc_norm_acc']*100) )
                        logger.add_scalars('train_iter_%s'%tgt, {'sign_classification':_Metric_['sgc_norm_acc']}, it+1)
                    else:
                        if tgt in _Metric_.keys():
                            Es = _Metric_[tgt].data.cpu().numpy().copy()
                            mean  = np.mean(Es)
                            median= np.median(Es)
                            rmse  = np.sqrt(np.mean(np.power(Es,2)))
                            acc11 = np.mean(Es < 11.25) * 100
                            acc22 = np.mean(Es < 22.5 ) * 100
                            acc30 = np.mean(Es < 30   ) * 100
                            logprint( '   %-10s  [mean]: %5.1f   [median]: %5.1f   [rmse]: %5.1f   [acc{11,22,30}]: %5.1f%%, %5.1f%%, %5.1f%%' % (tgt, mean, median, rmse,acc11,acc22,acc30) )
                            logger.add_scalars('train_iter_%s'%tgt, {'Mean':mean, 'Median':median, 'RMSE':rmse, 'Acc11':acc11, 'Acc22':acc22, 'Acc30':acc30}, it+1)

                print ("\r[work_dir] %s   B=%s \r" % (os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:], batch_size), end='', flush=True)

        # Do test
        if (epoch+1) % opt.test_step_epoch==0   or _is_last_epoch:
            mean, median, rmse,acc11,acc22,acc30 = test(dataset_test, work_dir, model, train_epoch=epoch+1)
            logger.add_scalars('test_epoch', {'Mean':mean, 'Median':median, 'RMSE':rmse, 'Acc11':acc11, 'Acc22':acc22, 'Acc30':acc30}, epoch+1)

        # Save model
        if (epoch+1)%opt.snapshot_step_epoch==0 or _is_last_epoch:
            save_checkpoint({
                'nr_adjust_lr': opt.nr_adjust_lr,
                'nr_it_and_epoch': (it+1, epoch+1),
                'state_dict': get_stripped_DataParallel_state_dict(model),
                'optimizer' : optimizer.state_dict(),
            }, filename=os.path.join(work_dir,'train_epoch_%s.pth.tar'%(epoch+1)))
        #
        epoch += 1   # 0 based

    logger.close()

if __name__ == '__main__':
    if opt.test_only:
        dataset_test  = _dataset_module( collection='test' , **_dataset_kwargs)
        test(dataset_test, work_dir, save_pred=opt.test_save_pred)
    else:
        train()

