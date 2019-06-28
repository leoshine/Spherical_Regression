# coding: utf-8
"""
 @Author  : Shuai Liao
"""

import os, sys
from basic.common import split_quoted, is_py3
from easydict import EasyDict as edict
from collections import OrderedDict as odict
import shlex
if not is_py3:
    from itertools import izip as zip

import numpy as np
import numpy.lib.recfunctions as rfn

"""
Think about:
1. How to read txt_table as numpy array.
2. How to handle None field.
3. How to handle string length.
"""

import re
""" ==================  regex 1 min cheat sheet  ==================
()  :  group you want to parse  (will be returned by match)
[]  :  a set of characters.
*   :  match 0 or more
+   :  match 1 or more
$   :  match the end   of the string
^   :  match the start of the string
.   :  match any character except line terminators like \n.
----------------[ examples ]-----------------
    [sfd]*   match 0 or more char that is in ['s','f','d'].
    \d+      match 1 or more numbers in [0-9]
    .\d+     match 1 or more numbers in [0-9] that followed by '.'
----------------[ Usage ]-----------------
pattern = re.compile('regex_expression')
match   = pattern.match('string_to_match')
if match:  # match is not None
    print match.groups()
else:
    print 'Matching failed. %s' % str_rep

"""


bbox = [323.,291.,342.,307.]
def bbox2str(bbox):
    x1,y1,x2,y2 = bbox
    return '"%3.1f %3.1f %3.1f %3.1f"' % (x1,y1,x2,y2)
    # ''.join(['%.1f'%x for x in bbox]) # .join()

raw_pattern   = '({[a-zA-Z0-9_@/.,()\[\]\*\+\-]*:[<^>]*\d+[.\d+]*[sfd]*})'
# if matches, return 4 groups: align, width, prec, typechar # correspond to 4 ()s
named_pattern = '{(?P<name>[a-zA-Z0-9_@/.,()\[\]\*\+\-]*):(?P<align>[<^>]*)(?P<width>\d+)(?P<prec>.\d+)*(?P<typechar>[sfd]*)}'


# def noneAs(A, a):
#     # if A is None, return a. Otherwise return A.
#     return A if A else a
#     # if A: return A
#     # else: return a

class FormatHandler:
    _type2typechar = {int:'d', float:'f', str:'s', }
    _typechar2type = {'d':int, 'f':float, 's':str, }

    def __init__(self, fmtstr ):
        self.fmtstr = fmtstr

        pattern = re.compile(named_pattern)
        match   = pattern.match(fmtstr)
        if match:  # match is not None
            self.fmt = edict(fmtstr=fmtstr, **match.groupdict())
        else:
            print ('Matching failed. %s' % fmtstr)
        #
        self.fmtstr_normal      = '{:%s%s%s%s}' % (self.fmt.align    if self.fmt.align     else '',
                                                   self.fmt.width    if self.fmt.width     else '',
                                                   self.fmt.prec     if self.fmt.prec      else '',
                                                   self.fmt.typechar if self.fmt.typechar  else '',)
        self.fmtstr_placeholder = '{:^%ss}' % self.fmt.width
        self._type_ = self._typechar2type[self.fmt.typechar]
        self.name = self.fmt.name

    # encode to string
    def __call__(self, value):
        try:
            return self.fmtstr_normal.format(value)
        except:
            return self.fmtstr_placeholder.format(value)
            # print self.fmtstr_placeholder, value

    def decode(self, str_rep):
        try   : return self._type_(str_rep)
        except: return None
        # return self._type_(str_rep)
        # if str_rep=='-':
        #     return None
        # else: return self._type_(str_rep)

class TxtTable:
    @staticmethod
    def parsefmt(fmt_str):
        name2fmter    = odict()
        # groupkeys = ['align', 'width', 'prec', 'typechar']
        # Find ['{img_id:<20s}', '{conf:<6.3f}', '{bbox:30s}', '{a:<4.1f}', '{e:<4.1f}', '{t:<4.1f}']
        fmtstr_list = re.findall(raw_pattern, fmt_str) # return
        # print fmtstr_list

        indexed_fmt_str = fmt_str
        for i, fmtstr in enumerate(fmtstr_list):
            indexed_fmt_str = indexed_fmt_str.replace(fmtstr, "{%d}"%i)

        for fmtstr in fmtstr_list:
            fmter = FormatHandler(fmtstr)
            name2fmter[fmter.fmt.name] = fmter
        return name2fmter, indexed_fmt_str


    def __init__(self, line_fmtstr=None, quoted=False):
        # "{img_id:<20s}  {conf:<6.3f}  {bbox:30s} {a:<4.1f}   {e:<4.1f}   {t:<4.1f}  "):
        # if quoted,  allows the using " x xx"  to include space in one split, but can be slower than str.split
        self.reset(line_fmtstr)
        #
        self.split_line_func = split_quoted   if quoted   else str.split

    def reset(self, line_fmtstr):
        self.line_fmtstr = line_fmtstr
        if line_fmtstr is not None:
            self.name2fmter, self.indexed_fmt_str = self.parsefmt(line_fmtstr)
            self.fields  = list(self.name2fmter.keys())
            self._fmters = list(self.name2fmter.values())

    def format(self, *values):
        assert len(values)==len(self.name2fmter.keys()), "Format is expecting %s params, but %s params received." % (len(self.name2fmter.keys()), len(values))
        return self.indexed_fmt_str.format(*[fmter(v) for fmter,v in zip(self._fmters, values)])
        # return self.indexed_fmt_str.format(*[fmter(v) for fmter,v in izip(self._fmters, values)])

    def getHeader(self):
        format_head = "# format: %s" % self.line_fmtstr
        fields_head = "# "+self.format(*self.name2fmter.keys())
        return format_head + '\n' + fields_head

    def parseline(self, entry_line):
        # sps = shlex.split(entry_line)  # Too slow!
        # sps = split_quoted(entry_line)
        # sps = entry_line.split()
        sps = self.split_line_func(entry_line)
        return dict([(fmt.name, fmt.decode(sp)) for fmt, sp in zip(self._fmters, sps)])
        # return dict([(fmt.name, fmt.decode(sp)) for fmt, sp in izip(self._fmters, sps)])
        # record = odict()
        # for name, sp in zip(self.fields, sps):
        #     record[name] = self.name2fmter[name].decode(sp)
        # return record
        # recode = []
        # for i in range(len(self._fmters)):
        #     recode.append(self._fmters[i].decode(sps[i]))
        # return recode
        # return [fmt.decode(sp) for fmt, sp in izip(self._fmters, sps)]
        # return dict(zip(self.fields, [fmt.decode(sp) for fmt, sp in izip(self._fmters, sps)]))  # 'zip' is time consuming.

    def loadrows(self, filename, fields=None, formatmarker='# format:'):
        """Return a list of row records."""
        lines = open(filename).readlines()
        #
        if self.line_fmtstr is None:
            assert lines[0].startswith(formatmarker), "[File format line not Found] : %s\n [InputFile]:" % (lines[0], filename)
            line_fmtstr = lines[0][len(formatmarker):].strip()
            # print lines[0].strip()
            self.reset(line_fmtstr)
        #
        _fields = fields if fields else self.fields  #
        _fmters = [self.name2fmter[x] for x in _fields]  # fields  = fields if fields else self.fields

        records = []
        for line in lines[1:]:
            line = line.strip()
            if not line.startswith('#'):
                # records.append( self.parseline(line) ) # avoid to extra func calling.
                # values = [fmt.decode(sp) for fmt, sp in izip(_fmters, line.split())]
                # records.append(values)
                # for fmt, sp in izip(_fmters, line.split()):
                for fmt, sp in zip(_fmters, line.split()):
                    records.append(fmt.decode(sp))
        return records

    def load(self, filename, fields=None, formatmarker='# format:'):
        """return an ordered dict of columns per each field"""
        lines = open(filename).readlines()
        #
        if self.line_fmtstr is None:
            assert lines[0].startswith(formatmarker), "[File format line not Found] : %s\n [InputFile]:" % (lines[0], filename)
            line_fmtstr = lines[0][len(formatmarker):].strip()
            # print lines[0].strip()
            self.reset(line_fmtstr)
        # after reset
        _fields  = fields if fields else self.fields  #
        _fmters  = [self.name2fmter[x]   for x in _fields]  # fields  = fields if fields else self.fields
        _colInds = [self.fields.index(x) for x in _fields]

        columns = [[] for _ in _fields] # []  # columns = odict([(x,[]) for x in self.fields])  #
        for line in lines[1:]:
            line = line.strip()
            if not line.startswith('#'):
                # sps = shlex.split(line)  # Too slow!
                # sps = split_quoted(line)
                # sps = line.split()
                sps = self.split_line_func(line)
                # for col, fmt, cInd in izip(columns, _fmters, _colInds):
                for col, fmt, cInd in zip(columns, _fmters, _colInds):
                    col.append(fmt.decode(sps[cInd]))
        return odict(zip(_fields, columns))
        # return odict(izip(_fields, columns))
                # # records.append( self.parseline(line) ) # avoid to extra func calling.
                # values = [fmt.decode(sp) for fmt, sp in izip(_fmters, line.split())]
                # # columns .append(records)
                # for i in range(len(columns)):
                #     columns[i].append(values[i])
        # return columns

    def load_as_recarr(self, filename, fields=None, formatmarker='# format:'):
        """Warning: if there's missing value, it will be filled with default value.
            See: https://docs.scipy.org/doc/numpy/user/basics.rec.html
            numpy.lib.recfunctions.merge_arrays()
                -1 for integers
                -1.0 for floating point numbers
                '-' for characters
                '-1' for strings
                True for boolean values
        """
        cols = self.load(filename, fields=fields, formatmarker=formatmarker)
        _fields, cols_data = cols.keys(), cols.values()  # None fields is updated during self.load
        cols_type = [self.name2fmter[field_name]._type_ for field_name in _fields]
        #for name,x,t in zip(fields, cols_data, cols_type):
        #    print '***', name,x,t
        cols_nparr = [np.array(x, dtype=t) for name,x,t in zip(_fields, cols_data, cols_type)]
        unnamed_recarr = rfn.merge_arrays(cols_nparr, flatten=True, usemask=False).view(np.recarray)
        return rfn.rename_fields(unnamed_recarr, dict(zip(unnamed_recarr.dtype.names, _fields)))

def test():
    txtTbl = TxtTable('{img_id:<20s}  {conf:<6.3f}  {bbox:30s}  {a:<4.1f}   {e:<4.1f}   {t:<4.1f}')
    print (txtTbl.getHeader())
    print (txtTbl.format('2011_003254', -2.345, bbox2str(bbox), 12, -40, 9.))
    print (txtTbl.format('2011_003254', -2.345, bbox2str(bbox), 12, '-', '-'))

    print (txtTbl.parseline('2011_003254            -2.345   323.0,291.0,342.0,307.0        12.0   -   -'))
    print (txtTbl.fields)

def test2():
    txtTbl = TxtTable('{cate:<20s}   {MedError:>6.3f}   {Acc@pi/6:>6.3f}   {Acc@pi/12:>6.3f}  {Acc@pi/24:>6.3f}')
    print (txtTbl.getHeader())
    print (txtTbl.format('boat', 2.345, 0.8, 0.6, 0.3))


if __name__ == '__main__':
    test()
