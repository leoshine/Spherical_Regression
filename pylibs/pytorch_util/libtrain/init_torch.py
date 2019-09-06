import os, sys
from basic.common import env, Open, add_path # rdict
import numpy as np
import math

import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


_cfg = edict( caffemodel=edict(),  # To be define later
              torchmodel=edict(),  # To be define later
              PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]]),
              # PIXEL_MEANS_Imagenet = np.array([[[104.006987932, 116.668767617, 122.678914341]]]),
            )
cfg = _cfg

this_dir = os.path.realpath(os.path.dirname(__file__))
base_dir = os.path.realpath(this_dir+'/../pretrained_model.cache')

# default models and pretrained weights.
cfg.caffemodel.alexnet = edict(
    proto = os.path.join( base_dir + '/bvlc_alexnet/deploy.prototxt'),
    model = os.path.join( base_dir + '/bvlc_alexnet/bvlc_alexnet.caffemodel'),
    pkl   = os.path.join( base_dir + '/bvlc_alexnet/bvlc_alexnet.pkl'),
    input_size = (227,227),
)
cfg.caffemodel.caffenet = edict(
    proto = os.path.join( base_dir + '/bvlc_reference_caffenet/deploy.prototxt'),
    model = os.path.join( base_dir + '/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'),
    pkl   = os.path.join( base_dir + '/bvlc_reference_caffenet/bvlc_reference_caffenet.pkl'),
    input_size = (227,227),
)
cfg.caffemodel.vgg16 = edict(
    proto = os.path.join( base_dir + '/vgg_net/VGG_ILSVRC_16_layers_deploy.prototxt'),
    model = os.path.join( base_dir + '/vgg_net/VGG_ILSVRC_16_layers.caffemodel'),
    pkl   = os.path.join( base_dir + '/vgg_net/VGG_ILSVRC_16_layers.pkl'),
    input_size = (224,224),
)
cfg.caffemodel.vgg19 = edict(
    proto = os.path.join( base_dir + '/vgg_net/VGG_ILSVRC_19_layers_deploy.prototxt'),
    model = os.path.join( base_dir + '/vgg_net/VGG_ILSVRC_19_layers.caffemodel'),
    pkl   = os.path.join( base_dir + '/vgg_net/VGG_ILSVRC_19_layers.pkl'),
    input_size = (224,224),
)
cfg.caffemodel.GoogLeNet = edict(
    proto = os.path.join( base_dir + '/bvlc_googlenet/deploy.prototxt'),
    model = os.path.join( base_dir + '/bvlc_googlenet/bvlc_googlenet.caffemodel'),
    input_size = (224,224),
)
cfg.caffemodel.vggm = edict(
    proto = os.path.join( base_dir + '/vgg_net/VGG_CNN_M_deploy.prototxt'),
    model = os.path.join( base_dir + '/vgg_net/VGG_CNN_M.caffemodel'),
    pkl   = os.path.join( base_dir + '/vgg_net/VGG_CNN_M.pkl'),
    input_size = (224,224),
)
cfg.caffemodel.resnet50 = edict(
    model = os.path.join( base_dir + '/resnet50-caffe.pth'),
)
cfg.caffemodel.resnet101= edict(
    model = os.path.join( base_dir + '/resnet101-caffe.pth'),
)
cfg.caffemodel.resnet152= edict(
    model = os.path.join( base_dir + '/resnet152-caffe.pth'),
)
# cfg.caffemodel.vgg16    = edict(
#     model = os.path.join( base_dir + '/vgg16_caffe.pth'),
# )
#--------------------------------------------------------- [torchmodel]
cfg.torchmodel.alexnet = edict(
    # module = torchvision.models.alexnet,
    model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    model   = os.path.join( env.Home, '.torch/models/alexnet-owt-4df8aa71.pth'),
    input_size = (224,224),
)
cfg.torchmodel.inception_v3_google = edict(
    # module = torchvision.models.alexnet,
    model  = os.path.join( env.Home, '.torch/models/inception_v3_google-1a9a5a14.pth'),
    input_size = (224,224),
)
cfg.torchmodel.resnet101 = edict(
    # module = torchvision.models.alexnet,
    model  = os.path.join( env.Home, '.torch/models/resnet101-5d3b4d8f.pth'),
    # input_size = (224,224),
)
cfg.torchmodel.vgg16 = edict(
    model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # model  = os.path.join( env.Home, '.torch/models/resnet101-5d3b4d8f.pth'),
    # input_size = (224,224),
)
cfg.torchmodel.vgg19 = edict(
    model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    # model  = os.path.join( env.Home, '.torch/models/resnet101-5d3b4d8f.pth'),
    # input_size = (224,224),
)
cfg.torchmodel.vgg16_bn = edict(
    model_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    # model  = os.path.join( env.Home, '.torch/models/resnet101-5d3b4d8f.pth'),
    # input_size = (224,224),
)
cfg.torchmodel.vgg19_bn = edict(
    model_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # model  = os.path.join( env.Home, '.torch/models/resnet101-5d3b4d8f.pth'),
    # input_size = (224,224),
)



def list_models(snapshots_dir, marker='iter'):
    import re
    # collect ".solverstate" and ".caffemodel" files.
    _files = os.listdir(snapshots_dir)

    reg_valid_name_str = '([()\[\]a-zA-Z0-9\s\.,_+-]+)'  # train_iter_20000.pth.tar
    reg_model = r"^%s_%s_(\d+).pth.tar$"  % (reg_valid_name_str, marker)
    # reg_state = r"^%s_iter_(\d+).solverstate$" % reg_valid_name_str

    net_names_model = set()
    iter_num_model = []
    model_files = [x for x in _files if x.endswith('.pth.tar') ]
    for name in model_files:
        match = re.search(reg_model, name)
        if match: # match is not None
            gs = match.groups()
        else:
            raise Exception('No model matches in snapshots_dir: %s\n\tExist model list: %s' % (snapshots_dir,model_files))
        assert len(gs)==2, '[Exception] in matching .pth.tar file:  "%s"' % name
        net_name, iter_num = gs[0], int(gs[1])
        net_names_model.add(net_name)
        iter_num_model.append(iter_num)
    assert len(net_names_model)==1, "None or Multiple net models in this dir: %s " % len(net_names_model)
    assert len(set(iter_num_model))==len(iter_num_model)
    iter_num_model.sort()

    existed_nums = iter_num_model
    net_name = list(net_names_model)[0]
    return existed_nums, net_name


def rm_models(snapshots_dir, type='KEEP_LATEST', marker='iter'):
    saved_nums, net_name = list_models(snapshots_dir, marker=marker) # ('snapshots')
    assert len(saved_nums)>0, "No models available"
    if type=='KEEP_LATEST':
        latest_model_name = '%s_%s_%s.pth.tar' % (net_name, marker, saved_nums[-1])  # '%s_iter_%s.pth.tar'
        latest_model_path = os.path.join(snapshots_dir, latest_model_name)
        assert os.path.exists(latest_model_path), latest_model_path
        if len(saved_nums)==1:  # only has latest model.
            return latest_model_name
        else:
            for it in saved_nums[:-1]:
                model_prefix = os.path.join(snapshots_dir, '%s_%s_%s') % (net_name, marker, it)
                model_path  = model_prefix+'.pth.tar'  # '.caffemodel'
                assert os.path.exists(model_path)
                os.system('rm -f %s' % (model_path))
            return latest_model_name
    else:
        raise NotImplementedError



def get_weights_from_caffesnapeshot(proto_file, model_file):
    import caffe
    from collections import OrderedDict
    import cPickle as pickle

    caffe.set_mode_cpu()

    net = caffe.Net(proto_file, model_file, caffe.TEST)

    model_dict = OrderedDict()
    for layer_name, param in net.params.iteritems():    # Most param blob has w and b, but for PReLU there's only w.
        learnable_weight = [] # {}
        if len(param)==2:
            # learnable_weight['w'] = param[0].data.copy()
            # learnable_weight['b'] = param[1].data.copy()
            learnable_weight.append(param[0].data.copy())
            learnable_weight.append(param[1].data.copy())
        elif len(param)==1:
            # learnable_weight['w'] = param[0].data.copy()
            learnable_weight.append(param[0].data.copy())
        else:
            raise NotImplementedError
        model_dict[layer_name] = learnable_weight

    return model_dict


def _copy_weights_from_caffemodel( own_state, pretrained_type='alexnet', ignore_missing_dst=False,
                  src2dsts = dict(conv1='conv1', conv2='conv2', conv3='conv3',
                                  conv4='conv4', conv5='conv5', fc6='fc6', fc7='fc7') ):
    """ src2dsts = dict(conv1='conv1', conv2='conv2', conv3='conv3', conv4='conv4', conv5='conv5')
    Or in list:
        src2dsts = dict(conv1=['conv1'], conv2=['conv2'], conv3=['conv3'], conv4=['conv4'], conv5=['conv5'])
    """
    print ("-----------------------")
    print ("[Info] Copy from  %s " % pretrained_type)
    print ("-----------------------")
    if isinstance(pretrained_type, tuple):
        proto_file, model_file = pretrained_type
        pretrained_weights = get_weights_from_caffesnapeshot(proto_file, model_file)
    else:
        import pickle
        print('Loading: %s' % cfg.caffemodel[pretrained_type].pkl)
        pretrained_weights = pickle.load( open(cfg.caffemodel[pretrained_type].pkl, 'rb'), encoding='bytes')  #  'bytes' )

        print (pretrained_weights.keys())
        # print (list(pretrained_weights.keys())[0].decode())

    not_copied = list(own_state.keys())

    src_list = sorted(src2dsts.keys())
    for src in src_list:  # src2dsts.iteritems():
        dsts = src2dsts[src]
        if not isinstance(dsts, list):
            dsts = [dsts]
        w, b =  pretrained_weights[src.encode('utf-8')]
        w = torch.from_numpy(w) # cast as pytorch tensor
        b = torch.from_numpy(b) # cast as pytorch tensor
        # one src can be copied to multiple dsts
        for dst in dsts:
            if ignore_missing_dst and dst not in own_state.keys(): # net.params.keys():
                print ('%-20s  -->  %-20s   [ignored] Missing dst.' % (src, dst))
                continue
            print ('%-20s  -->  %-20s' % (src, dst))
            dst_w_name = '%s.weight' % dst
            dst_b_name = '%s.bias'   % dst
            assert dst_w_name in own_state.keys(), "[Error] %s not in %s" %(dst_w_name, own_state.keys())
            assert dst_b_name in own_state.keys(), "[Error] %s not in %s" %(dst_b_name, own_state.keys())
            #-- Copy w
            assert own_state[dst_w_name].shape==w.shape, '[%s] w: dest. %s != src. %s' %(dst_w_name, own_state[dst_w_name].shape, w.shape)
            own_state[dst_w_name].copy_(w)
            not_copied.remove(dst_w_name)
            #-- Copy b
            assert own_state[dst_b_name].shape==b.shape, '[%s] w: dest. %s != src. %s' %(dst_b_name, own_state[dst_b_name].shape, b.shape)
            own_state[dst_b_name].copy_(b)
            not_copied.remove(dst_b_name)

    for name in not_copied:
        if   name.endswith('.weight'):
            own_state[name].normal_(mean=0.0, std=0.005)  # guassian for fc std=0.005,  for conv std=0.01
            print ('%-20s  -->  %-20s' % ('[filler] gaussian005', name))
        elif name.endswith('.bias'):
            own_state[name].fill_(0)
            print ('%-20s  -->  %-20s' % ('[filler] 0', name))
        else:
            print ("Unknow parameter type: ", name)
            raise NotImplementedError
    print ("-----------------------")


def _copy_weights_from_torchmodel(own_state, pretrained_type='alexnet', strict=True, src2dsts=None):
    # wrapper for load_state_dict
    from torch.nn.parameter import Parameter
    print ("-----------------------")
    print ("[Info] Copy from  %s " % pretrained_type)
    print ("-----------------------")
    src_state = model_zoo.load_url(cfg.torchmodel[pretrained_type].model_url)
    not_copied = list(own_state.keys())
    # print(type(not_copied))

    if src2dsts is not None:
        for src, dsts in src2dsts.items():
            if not isinstance(dsts, list):
                dsts = [dsts]
            # one src can be copied to multiple dsts
            for dst in dsts:
                if dst in own_state.keys():
                    own_state[dst].copy_(src_state[src])
                    not_copied.remove(dst)
                    print ('%-20s  -->  %-20s' % (src, dst))
                else:
                    dst_w_name, src_w_name = '%s.weight' % dst, '%s.weight' % src
                    dst_b_name, src_b_name = '%s.bias'   % dst, '%s.bias'   % src
                    if (dst_w_name not in own_state.keys() or dst_b_name not in own_state.keys()) and not strict: # net.params.keys():
                        print ('%-20s  -->  %-20s   [ignored] Missing dst.' % (src, dst))
                        continue
                    print ('%-20s  -->  %-20s' % (src, dst))
                    #-- Copy w
                    assert own_state[dst_w_name].shape==src_state[src_w_name].shape, '[%s] w: dest. %s != src. %s' %(dst_w_name, own_state[dst_w_name].shape, src_state[src_w_name].shape)
                    own_state[dst_w_name].copy_(src_state[src_w_name])
                    not_copied.remove(dst_w_name)
                    #-- Copy b
                    assert own_state[dst_b_name].shape==src_state[src_b_name].shape, '[%s] w: dest. %s != src. %s' %(dst_b_name, own_state[dst_b_name].shape, src_state[src_b_name].shape)
                    own_state[dst_b_name].copy_(src_state[src_b_name])
                    not_copied.remove(dst_b_name)
    else:
        for name, param in src_state.items():
            if name in own_state:  # find in own parameter
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    print ('%-30s  -->  %-30s' % (name, name))
                    own_state[name].copy_(param)
                    not_copied.remove(name)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            else:
                print ('%-30s  -->  %-30s   [ignored] Missing dst.' % (name, name))

    for name in not_copied:
        if   name.endswith('.weight'):
            #-# torch.nn.init.xavier_normal(own_state[name])
            #-# print '%-30s  -->  %-30s' % ('[filler] xavier_normal', name)
            own_state[name].normal_(mean=0.0, std=0.005)  # guassian for fc std=0.005,  for conv std=0.01
            print ('%-20s  -->  %-20s' % ('[filler] gaussian005', name))
        elif name.endswith('.bias'):
            own_state[name].fill_(0)
            print ('%-30s  -->  %-30s' % ('[filler] 0', name))
        elif name.endswith('.running_mean') or name.endswith('.running_var') or name.endswith('num_batches_tracked'):
            print ('*************************** pass', name)
        else:
            print ("Unknow parameter type: ", name)
            raise NotImplementedError

    if strict:
        missing = set(own_state.keys()) - set(src_state.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    print ("-----------------------")



# Copy pretrained weights to net.
# By specify scrs, dsts layer names, this can prevent problem of
# "same layer name has different shape" in traditional weight copy.
def copy_weights(own_state, pretrained_type, **kwargs):
    ''' Usage example:
            copy_weights(own_state, 'torchmodel.alexnet',  strict=False)
            copy_weights(own_state, 'caffemodel.alexnet',  ignore_missing_dst=True, src2dsts={})
            copy_weights(own_state, '')
    '''
    if isinstance(pretrained_type,str):
        if   pretrained_type.startswith('torchmodel.'):
            pretrained_type = pretrained_type[11:] # remove header
            copy_func = _copy_weights_from_torchmodel
        elif pretrained_type.startswith('caffemodel.'):
            pretrained_type = pretrained_type[11:] # remove header
            copy_func = _copy_weights_from_caffemodel
        else:
            print ("Unkonw pretrained_type: ", pretrained_type)
            raise NotImplementedError
    elif isinstance(pretrained_type,tuple):
        copy_func = _copy_weights_from_caffemodel
    else:
        print ("Unkonw type(pretrained_type): ", type(pretrained_type))
        raise NotImplementedError

    copy_func(own_state, pretrained_type, **kwargs)




def init_weights_by_filling(nn_module_or_seq, gaussian_std=0.01, kaiming_normal=True, silent=False):
    """ Note: gaussian_std is fully connected layer (nn.Linear) only.
        For nn.Conv2d:
           If kaiming_normal is enable, nn.Conv2d is initialized by kaiming_normal.
           Otherwise, initialized based on kernel size.
    """
    if not silent:
        print('[init_weights_by_filling]  gaussian_std=%s   kaiming_normal=%s \n %s' % (gaussian_std, kaiming_normal, nn_module_or_seq))
    for name, m in nn_module_or_seq.named_modules():
        if isinstance(m, nn.Conv2d):
            if kaiming_normal:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 'relu'
            else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:  m.weight.data.fill_(1)
            if m.bias   is not None:  m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0, std=gaussian_std) # std=0.01)  # ('[filler] gaussian005', name)
            m.bias.data.zero_()
        # torch.nn.init.xavier_normal(own_state[name])
    return nn_module_or_seq


def count_parameters_all(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def count_parameters_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
