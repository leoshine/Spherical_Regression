"""
 @Author  : Shuai Liao
"""

#@@@CODE_SNIPPET@@@  _SNP_obj_struct
# MAX_PATH_LEN = 128

def fmt_listdef(lines_str, indent=8):
    str2type = dict(
        str     =   '(str,MAX_STRING_LEN)',
        int     =   'np.int32',
        float   =   'np.float32',
        int32   =   'np.int32',
        uint8   =   'np.uint8',
        float32 =   'np.float32',
        bool    =   'np.bool',
    )
    # python code line
    def code_line(line):
        line = line.strip()
        assert line[0]!='#'
        ind = line.find('#')
        if ind>0:
            valid_str, comm_str = line[:ind].strip(), line[ind:].strip()
        else:
            valid_str, comm_str = line.strip(), ''

        ind = valid_str.find('=')
        assert ind>0, valid_str
        field_name = valid_str[:ind].strip()
        _rest = valid_str[ind+1:].strip()
        # if len(rest_str.split())>1:  # contains ' ' :  have shape definition.

        _ind = _rest.find(',') # find first ','
        if _ind>0:  # have shape definition.
            data_type, data_shape = _rest[:_ind].strip(), _rest[_ind+1:].strip()
            data_type = str2type.get(data_type, data_type) # data_type if data_type not in str2type.keys() else str2type[data_type]
            _code_line = " "*indent + "(%-20s , %-20s, %-12s ),  %s" % ("'%s'" % field_name, data_type, data_shape, comm_str)
        else:
            data_type = _rest
            data_type = str2type.get(data_type, data_type)
            _code_line = " "*indent + "(%-20s , %-20s  %s ),  %s" % ("'%s'" % field_name, data_type, ' '*12, comm_str)
        return _code_line.rstrip()
    # parse comment line
    def comm_line(line):
        line = line.strip()
        assert line[0]=='#', line
        return " "*indent + line.strip()

    fmt_lines = []
    for line in lines_str.split('\n'):
        # print line
        if line.strip().startswith('#'):
            fmt_lines.append(comm_line(line))
        else:
            fmt_lines.append(code_line(line))
    return '\n'.join(fmt_lines)


def get_code_blocks(lines_str):
    lines = lines_str.split('\n')
    code_blocks = []

    i=0
    while True:
        while lines[i].strip() == '':
            i+=1
            if i>=len(lines):
                return code_blocks

        start = i
        # track current code block until there's an empty line.
        while lines[i].strip() != '':
            i+=1
            if i>=len(lines):
                code_block = lines[start:i]  # remove indentation
                code_blocks.append("\n".join(code_block))
                return code_blocks
        end = i
        code_block = lines[start:i]  # remove indentation
        code_blocks.append("\n".join(code_block))

def block2list_code(lines_str):
    lines = lines_str.split('\n')

    i=0
    while not lines[i].strip().startswith('class'):
        i+=1
        if i>=len(lines):
            print ('[Not a class code block]')
            return lines_str # not a class code lines.
    forewords = '\n'.join(lines[:i])


    _tmp = lines[i].strip()
    _ind = _tmp.find(':')
    class_name = _tmp[5:_ind].strip()
    list_code = fmt_listdef('\n'.join(lines[i+1:]))
    return '%s\n%s = [\n%s\n]\n' % (forewords, class_name, list_code)


if __name__=="__main__":
    #table(object_anno)
    ss = """\
    obj_id          str                 # <primary key>  format: obj_id  = {image_id}-{x1},{y1},{x2},{y2}  e.g. 2008_000251-24,14,410,245
    image_id        str                 # <foreign key>
    category        str                 # e.g. aeroplane, car, bike
    cad_idx         int
    bbox            np.int32      4     # xmin,ymin,xmax,ymax
    #=====viewpoint (camera model parameters)
    view            viewpoint
    #==Other annotation.
    difficult       int
    truncated       bool
    occluded        bool\
"""

    code_blocks = get_code_blocks(open('db_type.def.py').read())
    open('db_type.py','w').write('\n\n'.join( map(block2list_code, code_blocks[:-1])) ) # the last code_block is this  'if __name__=="__main__":'
    for bl in code_blocks:
        #print bl, '\n\n\n'
        print (block2list_code(bl), '\n\n')
