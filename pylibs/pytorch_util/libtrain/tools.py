import torch
from collections import OrderedDict

def get_stripped_DataParallel_state_dict(m, base_name='', newdict=OrderedDict()):
    """ strip 'module.' caused by DataParallel.
    """
    try:
        # Test if any child exist.
        m.children().next()
        # Pass test, has one or more children
        if isinstance(m, torch.nn.DataParallel):
            assert len([x for x in m.children()])==1, "DataParallel module should only have one child, namely, m.module"
            get_stripped_DataParallel_state_dict(m.module, base_name, newdict)
        else:
            for _name, _module in m.named_children():
                new_base_name = base_name+'.'+_name if base_name!='' else _name
                get_stripped_DataParallel_state_dict(_module, new_base_name, newdict)
        return newdict # if ended here, return newdict

    except StopIteration:
        # No children any more.
        assert not isinstance(m, torch.nn.DataParallel), 'Leaf Node cannot be "torch.nn.DataParallel" (since no children ==> no *.module )'
        for k, v in m.state_dict().items():
            new_k = base_name+'.'+k
            newdict[new_k] = v
        return newdict # if ended here, return newdict


def patch_saved_DataParallel_state_dict(old_odict):
    print ("[Warning]\tNot recommend to use for unknown modules!\n\t\t\tUnless you're sure 'module.' is called by DataParallel.")
    assert isinstance(old_odict, OrderedDict)

    new_odict=OrderedDict()
    for k, v in old_odict.items():
        ind = k.find('module.')
        if ind>=0:
            new_k = k.replace('module.','')
            new_odict[new_k] = v
            print ('\t[renaming]     %-40s   -->  %-40s' % (k, new_k))
        else:
            new_odict[k] = v
    return new_odict






def test0():
    from torchvision.models import alexnet
    model = alexnet()
    org_state_dict = model.state_dict().copy()
    print (org_state_dict.keys())

    # make feature parallel.
    # Test 1.
    model.features = torch.nn.DataParallel(model.features)

    # Test 2.
    # model = torch.nn.DataParallel(model)

    print (model.state_dict().keys())

    # print model.features.module[0].state_dict().keys() #.children().next()  # [x for x in model.features.module[0].children()]
    # print isinstance(model.features.module[0], torch.nn.DataParallel)
    # exit()

    print ('-------------------[patch_state_dict]')
    new_state_dict1 = patch_saved_DataParallel_state_dict(model.state_dict())
    print (new_state_dict1.keys())
    print ('-------------------')
    assert org_state_dict.keys()==new_state_dict1.keys()
    # exit()


    print ('-------------------[get_stripped_state_dict]')
    new_state_dict2 = get_stripped_DataParallel_state_dict(model)
    print (new_state_dict2.keys())
    print ('-------------------')
    assert org_state_dict.keys()==new_state_dict2.keys()
    for k in org_state_dict.keys():
        assert torch.equal(org_state_dict[k], new_state_dict2[k])


if __name__ == '__main__':
    test0()
