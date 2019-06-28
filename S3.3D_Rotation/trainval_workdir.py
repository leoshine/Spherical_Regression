"""
 @Author  : Shuai Liao
"""

#import matplotlib
#matplotlib.use('Agg')  # server mode.
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
from basic.common import Open, env, add_path, RefObj as rdict, argv2dict, is_py3
this_dir = os.path.dirname(os.path.realpath(__file__))
import pickle
#
from pytorch_util.libtrain.yaml_netconf import parse_yaml, import_module_v2
from pytorch_util.libtrain.tools        import get_stripped_DataParallel_state_dict, patch_saved_DataParallel_state_dict
from txt_table_v1 import TxtTable
#
from tensorboardX import SummaryWriter
from tqdm import tqdm

#===========  Parsing from working path =========
pwd      = os.getcwd()  #  Assume: $base_dir/S3.3D_Rotation/{MtdFamily}/{MtdType}
MtdFamily, MtdType  = pwd.split(os.sep)[-2:]
#================================================


#------- args from convenient run yaml ---------
# For the purpose that no need to specific each run (without argparse)
convenient_run_argv_yaml=\
'''
MtdFamily       : {MtdFamily}    # e.g. regQuatNet
MtdType         : {MtdType}      # e.g. reg_Direct, reg_Sexp, reg_Sflat
net_module      : {net_module}   # same as MtdType here, namely import from 'MtdType'.py

net_arch        : {net_arch}     # e.g. alexnet, vgg16
base_dir        : {base_dir}     # e.g. path/to/S3.3D_Rotation
LIB_DIR         : {base_dir}/lib

train_view      : 100V  # 20V #

work_dir        : './snapshots/{net_arch}'
nr_epoch        : 150
test_step_epoch : 10

this_dir        : {this_dir}
base_lr         : 0.001
'''.format(net_arch='alexnet',
           base_dir=this_dir,
           this_dir=pwd,
           #
           MtdFamily =MtdFamily,
           MtdType   =MtdType,
           net_module=MtdFamily,
           )
run_args = parse_yaml(convenient_run_argv_yaml)


#------- arg from argparse -----------------
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('conf_yml_file', default='', type=str, metavar='PATH',
                    help='path to conf_yml_file (default: none)')
# parser.add_argument('work_dir'     , default='', type=str, metavar='PATH',)
                 # help='path to work_dir (default: none)')
parser.add_argument('gpu_ids'   , default='', type=str, metavar='PATH',
                    help='e.g. 0  or 0,1,2,3')
parser.add_argument('--resume'     , action="store_true", default=False,
                    help='to resume by the last checkpoint')
parser.add_argument('--pretrain'   , default=None, type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD or Adam')
parser.add_argument('--test_only'   , action="store_true", default=False,
                    help='only do test once.')
_pargs, _rest = parser.parse_known_args() # parser.parse_args()
# parse the rest undefined args with "--key=value" form.
_cmd_args = argv2dict(_rest)
_cmd_args.update( vars(_pargs) )
#
run_args.update(_cmd_args)

from string import Template
template_str = open( os.path.join(this_dir, 'conf_template.yml')).read()
template = Template(template_str)
print(run_args)
conf_yml_str = template.substitute(run_args)

#-- parse module_yml_file
opt = parse_yaml(conf_yml_str )
#
opt.update( run_args )
#
from ordered_easydict import Ordered_EasyDict as oedict
opt = oedict(opt) # Use opt for reference all configurations.


#------ Import modules ----------
[(_dataset_module,_dataset_kwargs), netcfg] = import_module_v2(opt.IMPORT_dataset)  # pred2angle
[(_net_module,_net_kwargs) ] = import_module_v2(opt.IMPORT_makenet)  # [_net_type]
[(eval_cates,_), (compute_geo_dists,_)]             = import_module_v2(opt.IMPORT_eval.GTbox)
net_arch = opt.net_arch  # or _net_kwargs.net_arch

_cfg = netcfg[net_arch]  # [opt.net_arch]
np.random.seed(_cfg.RNG_SEED)
torch.manual_seed(_cfg.RNG_SEED)
if opt.use_gpu:
    torch.cuda.manual_seed(_cfg.RNG_SEED)

#---------------------------------------------------------------------------------------------------[dataset]
dataset_test  = _dataset_module( collection='test' , sampling=0.2, **_dataset_kwargs)   #
dataset_train = _dataset_module( collection='train', **_dataset_kwargs) #
# change the sampling of dataset: e.g. sampling: {imagenet:1.0, synthetic:1.0}
# 'ModelNet10/SO3_100V.white_BG_golden_FG'

# From default.run.conf.yml.sh
if opt.cates is None:
    opt.cates = dataset_train.cates
cates = opt.cates

if 'batch_size' in opt:
    batch_size = opt.batch_size
else:
    batch_size = _cfg.TRAIN.BATCH_SIZE
nr_GPUs = len(opt.gpu_ids)
assert nr_GPUs>=1, opt.gpu_ids
if nr_GPUs>1:
    print ('---------------------   Use multiple-GPU %s   -------------------------' % opt.gpu_ids)
    print ('     batch_size  = %s' % batch_size  ) #(batch_size*nr_GPUs)
    print ('     num_workers = %s' % (opt.num_workers*nr_GPUs))
#
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  # batch_size*nr_GPUs
                                           num_workers=opt.num_workers*nr_GPUs, pin_memory=opt.pin_memory, sampler=None)

#---------------------------------------------------------------------------------------------------[model]
print ('[makenet] nr_cate: ', len(cates))
model = _net_module(nr_cate=len(cates), **_net_kwargs)  #len(_cfg.cates))
if 'fix_conv1_conv2' in opt.keys() and opt.fix_conv1_conv2:
    model.fix_conv1_conv2()
#
watch_targets = model.targets

#---------------------------------------------------------------------------------------------------[optimizer]
params = []
for name, param in model.named_parameters():
    print ('----(*)  ', name)
    if param.requires_grad:
        params.append(param)

print('[Optimizer] %s' % opt.optimizer)
if   opt.optimizer=='Adam':
    optimizer = torch.optim.Adam(params, lr=opt.base_lr) # , weight_decay=opt.weight_decay)
elif opt.optimizer=='SGD':
    optimizer = torch.optim.SGD( params, opt.base_lr, # model.parameters(), opt.base_lr,
                                 momentum=opt.momentum,
                                 weight_decay=opt.weight_decay )
else:
    raise NotImplementedError



work_dir = opt.work_dir
work_dir += '.%s' % opt.train_view
_short_work_dir = os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:]

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
    print( "\r%s                            "%s)
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
        start_it, start_epoch = checkpoint['it_and_epoch']  # mainly for control lr: (it, epoch)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # fix for pytorch 4.0.x  [https://github.com/jwyang/faster-rcnn.pytorch/issues/222]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
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



# Check if use multi-gpus.
# Note should be after any "model.load_state_dict()" call!
if opt.use_gpu:
    # model.cuda()
    if nr_GPUs>1: # multi-GPUs   opt['mGPUs']:
        # see: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        if net_arch.startswith('alexnet'):
            model.trunk.Convs = torch.nn.DataParallel(model.trunk.Convs)
        elif net_arch.startswith('vgg'):
            model.trunk.features = torch.nn.DataParallel(model.trunk.features)
        else:
            model.trunk = torch.nn.DataParallel(model.trunk)
    model.cuda()


if not os.path.exists(work_dir):
    print ("[Make new dir] ", work_dir)
    os.makedirs(work_dir)



disp_interval = 10  if ('disp_interval' not in opt)  else opt['disp_interval']

"""
(nr_iter * batch_size)/nr_train = nr_epoch   where nr_train=28647/29786
 (40000*200)/29786. = 268.6
 (40000* 50)/29786. =  67.1
 (40000*_cfg.TRAIN.BATCH_SIZE)/29786. / 2 /10
"""
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every _N_ epochs"""
    _N_ = int((40000*batch_size)/29786./2 /10)*10  # '2' mean at most decay 2 times.
    lr = opt.base_lr * (0.1 ** (epoch // _N_)) # 300))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_by_iter(optimizer, cur_iter, max_iter):
    """Sets the learning rate to the initial LR decayed by 10 every _N_ epochs"""
    _N_ = max_iter // 3  # add just learning rate 3 times.
    lr = opt.base_lr * (0.1 ** (max(cur_iter,0) // _N_)) # 300))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_by_epoch(optimizer, cur_epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every _N_ epochs"""
    _N_ = max_epoch // 3  # add just learning rate 3 times.
    lr = opt.base_lr * (0.1 ** (max(cur_epoch,0) // _N_)) # 300))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



def test(dataset_test, work_dir, test_model=None, marker='epoch'):
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
    gLoss_redu = reducer_group(*watch_targets)
    gPred_redu = reducer_group(*['quat'])

    pre_time = time.time()
    it = -1
    epoch = -1
    #
    keys = dataset_test.keys
    test_loader = torch.utils.data.DataLoader( dataset_test, batch_size=_cfg.TEST.BATCH_SIZE*nr_GPUs, shuffle=False,
                                               num_workers=opt.num_workers*nr_GPUs, pin_memory=opt.pin_memory, sampler=None)

    with torch.no_grad():
        pbar = tqdm(test_loader)

        for _i_, sample_batched in enumerate(pbar):
            pbar.set_description("[work_dir] %s  " % _short_work_dir)
            it += 1

            # Note: Tensor.cuda()   Returns a copy of this object in CUDA memory.
            label = torch.autograd.Variable( sample_batched['label'].cuda(non_blocking=True) )
            data  = torch.autograd.Variable( sample_batched['data' ].cuda(non_blocking=True) )
            # formulate GT dict
            _gt_targets = test_model.gt_targets if hasattr(test_model, 'gt_targets') else test_model.targets
            GT    = edict()
            for tgt in _gt_targets:
                GT[tgt] = torch.autograd.Variable( sample_batched[tgt].cuda(non_blocking=True) )

            # compute Pred output
            Prob = test_model(data, label)

            # compute Loss for each target and formulate Loss dictionary.
            Loss = test_model.compute_loss(Prob, GT)

            total_loss = 0
            for tgt in watch_targets:
                total_loss += Loss[tgt]

            # predict target angles value
            Pred = test_model.compute_pred(Prob)

            gLoss_redu.collect(Loss) # pass in dict of all loss (loss_a, loss_e, loss_t).
            gPred_redu.collect(Pred, squeeze=False)

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

            # pbar.set_description("[work_dir] %s  " % os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:])
            # print ("\r[work_dir] %s \r" % os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:],end='')
            sys.stdout.flush()

    pred_quats = gPred_redu.reduce()['quat']


    #-- Write result to file  (Format: # {obj_id}  {a} {e} {t} )
    txtTbl = TxtTable('{obj_id:<20s}   {a:>6.4f}  {b:>6.4f}  {c:>6.4f}  {d:>6.4f}')
    rslt_lines = [ txtTbl.getHeader() ]
    for _k,_quat in zip(keys, pred_quats):
        _a,_b,_c,_d=_quat
        rslt_line = txtTbl.format(_k,_a,_b,_c,_d)
        rslt_lines.append(rslt_line)
    rslt_lines = '\n'.join(rslt_lines)
    Open(out_rslt_path,'w').write(rslt_lines)
    #
    print ('[out_rslt_path]', out_rslt_path)

    #-- Do evaluation  ('MedError', 'Acc@theta')
    from numpy_db import npy_table
    rc_tbl = npy_table(dataset_test.recs)
    #
    summary_str = eval_cates(out_rslt_path, rc_tbl, cates=opt.cates, theta_levels_str='pi/6  pi/12  pi/24') # ['aeroplane','boat','car'])
    Open(out_eval_path,'w').write(summary_str)
    print (summary_str)

    reca = TxtTable().load_as_recarr(out_eval_path, fields=['MedError', 'Acc@pi/6', 'Acc@pi/12', 'Acc@pi/24'])

    return reca[-1]



def train(nr_disp=5000):
    os.system('rm -rf %s ' % (work_dir+"/logs") )
    logger = SummaryWriter(work_dir+"/logs")
    nr_epoch = opt.nr_epoch
    nr_iter  = opt.nr_epoch * (len(dataset_train)/batch_size)  # 130800

    # based on iter
    disp_interval= int(nr_iter/nr_disp)

    pre_time = time.time()
    it    = start_it    -1 # -1
    epoch = start_epoch -1 # -1
    #
    while epoch<nr_epoch:
        epoch += 1
        # Do test first
        if epoch % opt.test_step_epoch==0:
            mederr, acc6, acc12, acc24 = test(dataset_test, work_dir, model)
            logger.add_scalars('acc/test', {'MedError':mederr, 'Acc@pi/6':acc6, 'Acc@pi/12':acc12, 'Acc@pi/24':acc24}, epoch+1)

        # switch to train mode
        model.train()
        lr = adjust_learning_rate_by_epoch(optimizer, epoch, nr_epoch)  # opt.base_lr

        pbar = tqdm(train_loader)
        for _i_, sample_batched in enumerate(pbar):
            pbar.set_description("[work_dir] %s   B=%s " % (_short_work_dir, batch_size))
            rec_inds = sample_batched['idx'].numpy()
            #
            it += 1
            label = torch.autograd.Variable( sample_batched['label'].cuda(non_blocking=True) )
            data  = torch.autograd.Variable( sample_batched['data' ].cuda(non_blocking=True) )
            # formulate GT dict
            _gt_targets = model.gt_targets if hasattr(model, 'gt_targets') else model.targets
            GT    = edict()
            for tgt in _gt_targets:
                GT[tgt] = torch.autograd.Variable( sample_batched[tgt].cuda(non_blocking=True) )

            # compute Pred output
            Prob = model(data, label)

            # compute Loss for each target and formulate Loss dictionary.
            Loss = model.compute_loss(Prob, GT)

            total_loss = 0
            for tgt in watch_targets:
                total_loss += Loss[tgt]  # * loss_weight

            # compute gradient and do SGD step
            optimizer.zero_grad()  # Clears the gradients of all optimized Variable s.
            total_loss.backward()

            optimizer.step()

            logger.add_scalars('loss_iter', Loss, it+1)
            # logger.add_scalar('grad_norm/fc7', fc7_gradNorm, it+1)


            # print loss info
            if it % disp_interval == 0: # or (it+1)==len(dataset_train)/batch_size:
                cur_time = time.time()
                time_consume = cur_time - pre_time
                pre_time = cur_time
                logprint( '%s [epoch] %3d/%3d [iter] %5d -----------------------------------[time_consume] %.2f   lr=%.8f'
                     % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), epoch+1, nr_epoch, it+1, time_consume, lr) )

                for tgt in watch_targets:
                    _loss = Loss[tgt].data.cpu().numpy().copy()
                    logprint( '  %-15s  loss=%.3f' % (tgt, _loss) )
                    if np.isnan(_loss):
                        print ("[Warning]  Weights explode!  Stop training ... ")
                        exit(-1)
                # Compute Acc@theta
                recs = dataset_train.recs[rec_inds]
                Pred = model.compute_pred(Prob)
                geo_dists = compute_geo_dists(Pred['quat'], recs.so3.quaternion)
                MedError = np.median(geo_dists) /np.pi*180.
                theta_levels = odict(zip(['pi/6','pi/12','pi/24'],[np.pi/6,np.pi/12,np.pi/24]))
                # # {'pi/6':np.pi/6, 'pi/12':np.pi/12, 'pi/24':np.pi/24})
                Acc_at_ts = odict([(tname,sum(geo_dists<tvalue)/float(len(geo_dists))) for tname,tvalue in theta_levels.items()])
                logger.add_scalars('acc/train', Acc_at_ts, it+1)
                acc_str = '   '.join(['[%s] %3.1f%%' %(k,Acc_at_ts[k]*100) for k,v in theta_levels.items()])
                logprint('  Acc@{ %s }   '%acc_str)
                # pbar.set_description("[work_dir] %s   B=%s \r" % (os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:], batch_size))
                # print ("\r[work_dir] %s   B=%s \r" % (os.path.abspath(work_dir)[len(os.path.abspath(opt.base_dir))+1:], batch_size), end='')
                sys.stdout.flush()
                #

        #
        if (epoch+1)%opt.snapshot_step_epoch==0:
            save_checkpoint({
                'it_and_epoch': (it, epoch),
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, filename=os.path.join(work_dir,'train_epoch_%s.pth.tar'%(epoch+1)))

    logger.close()

if __name__ == '__main__':
    if opt.test_only:
        test(dataset_test, work_dir)
    else:
        train()
        rm_models(work_dir, marker='epoch')

