# coding: utf-8
"""
 @Author  : Shuai Liao
"""

from __future__ import print_function
import sys
import os
import time, datetime
import platform
import re
from collections import OrderedDict as odict
from ast import literal_eval  # safe way to do eval('str_expression')
import numpy as np

os_name = platform.system()
is_mac   = (os_name=='Darwin')
is_linux = (os_name=='Linux' )

# One liner detect if it's ran by python3
is_py3 = (sys.version_info > (3, 0))

class Env(object):
    def __getattr__(self, name):
        return os.getenv(name)
env = Env()
# Example:
# from basic.common imoprt env
# print env.HOME

def argv2dict(sys_argv):
    """ Auto parsing  ---key=value like arguments into dict.
    """
    # if matches, return 2 groups: key, value # correspond to 2 ()s
    named_pattern_exp = '--(?P<key>[a-zA-Z0-9_-]*)=(?P<value>.*)'
    pattern = re.compile(named_pattern_exp)

    rslt = odict()
    for arg in sys_argv:
        match   = pattern.match(arg)
        if match:  # match is not None
            gd = match.groupdict()
            key = gd['key']
            try:# To test if 'value' string is a python expression.
                # it's a python expression, here can return float, int, list etc.
                rslt[key] = literal_eval( gd['value'] )
            except:
                # it's not a python expression, just return such string.
                rslt[key] = gd['value']
        else:
            # print('Matching failed. %s' % arg)
            print('  [argv2dict] skip: %s' % arg)
            pass
    return rslt


#
class RefObj(object):  # Reflection class
    def __init__(self, **kwargs):
        from collections import OrderedDict, Counter
        super(RefObj, self).__setattr__('vardict', OrderedDict())
        self.update(**kwargs)

    def __setattr__(self, name, value):
        self.vardict[name] = value

    def __getattr__(self, name):
        return self.vardict[name]

    def __setitem__(self, name, value):
        self.vardict[name] = value

    def __getitem__(self, name):
        return self.vardict[name]

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.vardict[name] = value

    # TODO
    # def todict(self):
    # def __str__(self): # use pprint

    def help(self):
        print('''
from basic.common import RefObj
rf = RefObj()
rf.hi = 'hello'
print rf.hi
print rf['hi']

It looks similar to easydict
https://pypi.python.org/pypi/easydict/
"EasyDict allows to access dict values as attributes (works recursively)."
from easydict import EasyDict as edict
rf = edict()
rf.hi = 'hello'
print(rf.hi)
''')
# make alias
rdict = RefObj


def Open(filepath, mode='r'): #, overwrite=1
    """ wrapper of open, auto make dirs when open new write file. """
    if mode.startswith('w') or mode.startswith('a'):
        try:
            os.makedirs( os.path.dirname(filepath) )
            print( "[Warning] Open func create dir: ", os.path.dirname(filepath) )
        except: pass
    return open(filepath, mode)


# Dynamically add PYTHONPATH for import
def add_path(*paths):
    for path in paths:
        path = os.path.abspath(path)
        assert os.path.exists(path), "[Warning] path not exist: %s" % path
        sys.path.insert(0, path)
        #if path not in sys.path:
        #    sys.path.insert(0, path)


def reload_safari():
    osascript_cmd = '''\
# reload current safari page
tell application "Safari"
    activate
    tell application "System Events"
        tell process "Safari"
            keystroke "r" using {command down}
        end tell
    end tell
end tell
# return focus back to opencv window
tell application "System Events" to tell process "python"
    set frontmost to true
    windows where title contains "show_image"
    if result is not {} then perform action "AXRaise" of item 1 of result
end tell'''
    os.system("osascript -e '%s'" % osascript_cmd)



def get_snippet_blocks(snippet_mark_tag, src_code):
    assert os.path.exists(src_code), "[Error] source code doesn't exit: %s " % src_code
    lines = open(src_code).readlines()
    code_blocks = []

    i=0
    while True:
        while lines[i].find('#@@@CODE_SNIPPET@@@')<0:
            i+=1
            if i>=len(lines):
                return "\n\n\n".join(code_blocks) if len(code_blocks)>0 else None
        if lines[i].strip().split()[-1]==snippet_mark_tag:
            start = i
            indentation = lines[i].find('#@@@CODE_SNIPPET@@@')
        else:
            i+=1
            continue
        # track current code block until there's an empty line.
        while lines[i].strip() != '':
            i+=1
            if i>=len(lines):
                code_block = [x[indentation:] for x in lines[start:i]] # remove indentation
                code_blocks.append("".join(code_block))
                return "\n\n\n".join(code_blocks) if len(code_blocks)>0 else None
        end = i
        code_block = [x[indentation:] for x in lines[start:end+1]] # remove indentation
        code_blocks.append("".join(code_block))
def snippet_code(snippet_mark_tags, context_lines, src_code_path=None):
    assert src_code_path is not None, "[Error] No src_code_path! Use src_code_path=__file__ to snippet current script."
    snp_script = "\n".join(context_lines)+'\n\n'
    for tag in snippet_mark_tags:
        snp_script += get_snippet_blocks(tag, src_code=src_code_path)
        snp_script += '\n\n\n'
    return snp_script
"""
Example Usage:
    #@@@CODE_SNIPPET@@@  _SNP_example_load
    # Loading example
    ObjIDs, Objs = pickle.load(file(outfile))
    ObjsID2Ind = dict(zip(ObjIDs, range(len(ObjIDs))))

    # Write a txt file for the usage of this pickled data.
    snp_str = \
    snippet_code( ['_SNP_obj_struct'], context_lines=['import cPickle as pickle', 'outfile="%s"'%outfile],
                  src_code_path=__file__)
    with open(outfile[:-4]+'.txt','w') as f:
        f.write(snp_str)
"""


def parse_yaml(conf_yaml_str):
    import yaml
    from easydict import EasyDict as edict
    print( '---------------Parsing yaml------------------' )
    _opt = yaml.load(conf_yaml_str) #
    opt = edict(_opt)
    return opt

def split_quoted(s):
    # See discussion: https://stackoverflow.com/questions/79968/split-a-string-by-spaces-preserving-quoted-substrings-in-python
    sp = re.findall(r'"[^"]+"|\'[^\']+\'|[^"\'\s]+',s)
    return [x.strip(' "') for x in sp]


def set_common_unique(set1, set2):
    """
       Input :
         Two sets: set1, set2 (can be any iterable object.)
       Output:
         common:  intersection(set1, set2)
         unique1:  set1-set2
         unique1:  set2-set1
    """
    union = []
    unique1, unique2 = set(set1), set(set2)
    _set2 = set(set2)
    for it in set1: # tqdm(
        if it in _set2:
            union.append(it)
            unique1.remove(it)
            unique2.remove(it)
    return union, unique1, unique2

##"""
##========================================================================================================================
###///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
##------------------------------------------------------------------------------------------------------------------------
##Statement
##*** This part is:
##***     Common tools forged by Shine
##------------------------------------------------------------------------------------------------------------------------
###\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
##========================================================================================================================
##"""
### The newPath is a folder path
### if it's not existed, make the newPath folder recursively
##def mkdirs(newPath):
##    if os.path.isdir( newPath ):
##        return
##    else:
##        parentPath = os.path.dirname(newPath)
##        if os.path.isdir( parentPath ):
##            os.makedirs( newPath )
##        else:
##            mkdirs( parentPath )
##            os.makedirs( newPath )

### The newPath is a file
### if it's not existed, then make the parent folder for the file recursivley.
##def mkdir4file(newfilePath):
##    if os.path.isfile( newfilePath ):
##        return
##    else:
##        parentPath = os.path.dirname(newfilePath)
##        mkdirs(parentPath)

def mkdir4file(newfilePath):
    makedirsforfile(newfilePath)


def auto_time_fmt(secs):
    if secs>60:
        return str(datetime.timedelta(seconds=secs))
    else:
        return '%.3f sec' % secs
# similar to timer
class Checkpoint(object):
    def __init__(self, debug=False):
        self.is_debug = debug
        self.reset()
        self.tic()

    def reset(self):
        if not self.is_debug:
            self.total_time = 0.
            self.start_time = 0.
            self.diff       = 0.

    def tic(self):
        if not self.is_debug:
            # using time.time instead of time.clock because time time.clock
            # does not normalize for multithreading
            self.start_time = time.time()

    def toc(self):
        if not self.is_debug:
            self.diff = time.time() - self.start_time
            self.total_time += self.diff
            return self.diff

    def now(self, reason=''):
        if not self.is_debug:
            if self.start_time!=0:
                print( '    {reason:50s}   {time:5s}'.format(time=auto_time_fmt(self.toc()),
                                    reason=reason)
                )
            self.tic()

    def ttl(self, reason=''):
        if not self.is_debug:
                print( '    {reason:50s}   {time:5s}'.format(time=auto_time_fmt(self.total_time),
                                    reason='[total] %s' % reason)
                )
class Timer(object):
    """A simple timer."""
    def __init__(self, reason='', debug=True):
        self.reason = reason
        self.debug  = debug
        #
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    # As controlled environment obj
    def __enter__(self):
        if self.debug:
            self.start_time = time.time()

    # As controlled environment obj
    def __exit__(self, type, value, traceback):
        if self.debug:
            diff = time.time() - self.start_time
            print( '    [Timer] {reason:50s}   {time:5s}'.format(time=auto_time_fmt(diff), reason=self.reason) )

'''  # Example
with Timer('Message xxx:', debug=True) as ct:
    # Do something
    # ...
    # On exit, the message + time cost will be auto printed.
'''

def timestamp_datetime(value,format = '%Y-%m-%dT%H:%M:%SZ'):
    #format = '%Y-%m-%dT%H:%M:%SZ' #'%Y-%m-%d %H:%M:%S'
    # value is the integer time stamp passed in, e.g. 1332888820
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def datetime_timestamp(dt):
    # In my Linux 32 bit computer, the boundaries are within signed 32bit integer,
    # so a tuple that translates to less than -2147483648 (ie. before (1901, 12, 13, 19, 44, 16))
    # or past 2147483647, or (2038, 1, 19, 3, 14, 7), will trigger that exception...
    # mine is 1971,3001
    splitIdx = dt.find('-')
    year = int(dt[:splitIdx])
    if year<1971 or year>3001:
        print( '\n****** Year excceed range of [1971,3001] detected, -> %d**'%year )
        print( '****** using fake year(1971 or 3001) in replace instead.****\n')
        if year<1971:
            dt = '1971'+dt[splitIdx:]
        else:
            dt = '3001'+dt[splitIdx:]

    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return float(s)  #return int(s)


import matplotlib.pyplot as plt
def quit_figure(event):
    # print("event.key", event.key)
    if   event.key == 'q':
        plt.close(event.canvas.figure)
    elif event.key == 'escape':
        plt.close(event.canvas.figure)
        exit()
#
def plt_wait(delay=None, event_func=quit_figure): # delay: in millisecond.
    if delay is not None:
        plt.tight_layout()
        plt.draw() #
        plt.ioff()
        plt.waitforbuttonpress(delay/1000) # 1e-4) # 0.00001)
    else:
        cid = plt.gcf().canvas.mpl_connect('key_press_event', event_func)
        plt.tight_layout()
        plt.show() #(block=False)

def plt_save(file_name, dpi=None, transparent=False):
    plt.tight_layout()
    plt.savefig(file_name, dpi=None, transparent=transparent)

import cv2
def cv2_wait(delay=-1):  # delay: in millisecond.
    key = cv2.waitKey(delay) & 0xFF
    if key==27:          # Esc key to stop
        cv2.destroyAllWindows()
        exit()
    return key

def cv2_putText(image, left_bottom, display_txt, bgcolor=None, fgcolor=(255,0,0), scale=0.5,thickness=1):
    xmin, ymax = left_bottom  # Note: here xmin, ymax is of the text box
    fontface, fontscale, color, thickness = cv2.FONT_HERSHEY_DUPLEX, scale, fgcolor, thickness
    if bgcolor is not None:
        size = cv2.getTextSize(display_txt, fontface, fontscale, thickness)[0]
        top_left, bottom_right = (xmin, ymax-size[1]), (xmin+size[0], ymax+5) # for ymin and 5 more pixel
        cv2.rectangle(image, top_left, bottom_right, bgcolor, cv2.FILLED)     # opencv2 CV_FILLED
    cv2.putText(image, display_txt, (xmin, ymax), fontface, fontscale, color,thickness,lineType=cv2.LINE_AA) # opencv2 cv2.CV_AA


import linecache
class file_linecache:
    def __init__(self, file_name, start_line_no=1):
        assert os.path.exists(file_name), '[File not found] %s' % file_name
        self.file_name = file_name
        self.cur_line_no = start_line_no # (Notice that linecache counts from 1)

    def getline(self):
        line = linecache.getline(self.file_name, self.cur_line_no)
        if line=='': # end of file or none existed line_no
            line = None
        else:
            self.cur_line_no += 1
        return line
