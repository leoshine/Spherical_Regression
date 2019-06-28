import os, sys
# from cafferas import list_models, copy_weights
from basic.common import add_path
from basic.util   import parse_yaml, dump_yaml, load_yaml
# import yaml
# import oyaml as yaml  # oyaml is a drop-in replacement for PyYAML which preserves dict ordering.
from pprint import pprint
# from ordered_easydict import Ordered_EasyDict as oedict  # my module
import json

# def parse_yaml(conf_yaml_str):
#     # from easydict import EasyDict as edict
#     print ('---------------Parsing yaml------------------[oyaml]')
#     _opt = yaml.load(conf_yaml_str) #
#     return _opt  # as Ordered_Dict
#     # opt = oedict(_opt)
#     # return opt
#
# def dump_yaml(odict, filepath=None):
#     _opt = yaml.dump(odict) #
#     if filepath is not None:
#         with open(filepath, 'w') as f:
#             f.write(_opt)

# for checking the loaded yaml conf.
def print_odict(odict):
    print(json.dumps(odict, indent=2, separators=(',', ': ')))

from importlib import import_module as _import_module
def import_module(info):
    # for module_name, info in module_infos:
    print ("[import_module] ", info['from'])
    if 'path' in info:
        add_path(info['path'])
        print ('  add_path: ', info['path'])
    mod = _import_module(info['from'])

    if 'import' in info:
        comps = []
        for comp in info['import']:
            comps.append( getattr(mod, comp)  )
        return comps
    else:
        return mod
    # return getattr(mod, 'DataEngine') # __getattr__['DataEngine']
    # DataEngine, get_anno, _cfg
    # return


def import_module_v2(info):
    # for module_name, info in module_infos:
    print( "[import_module] ", info['from'])
    if 'path' in info:
        add_path(info['path'])
        print ('  add_path: ', info['path'])
    mod = _import_module(info['from'])

    if 'import' in info:
        comps = []
        for comp, kwargs in info['import'].items():
            # print '>>>>', comp, kwargs
            try:
                if kwargs is None:  # comp is variable
                    _var = getattr(mod, comp)
                    comps.append( _var  )
                else:  # comp is function with kwargs
                    _func = getattr(mod, comp)
                    comps.append( (_func, kwargs))  # (_func(**kwargs))  #
            except Exception as inst:
                print('\n[Exception] %s' % inst)
                pprint(dict(info)) # (comp, kwargs)
        return comps
    else:
        return mod

