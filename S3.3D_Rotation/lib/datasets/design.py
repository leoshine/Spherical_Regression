import numpy as np



def test0():
    nr_bin = 21   # bin_size = 17.14   (360./21)
    bin_size = 360./nr_bin
    print 'bin_size: ', bin_size

    def intervals():
        step  = bin_size
        start = bin_size/2.  # start from half bin size
        stop  = 360.-(bin_size/2.) +1e-2 # +1e-2 just because np.arange will exclude right boundary.
        return np.arange(start, stop, step)
    print [int(x*10)/10. for x in intervals()]

    def bin_a(a):
        return ( np.int64( np.floor(a*nr_bin/360.    )) )  %nr_bin

    def bin_t(t):
        return ( np.int64( np.floor(t*nr_bin/360.+9.5)) ) %nr_bin

    a = 17.1 # 8.5
    print  'a %4.1f  -->   %4d' % (a, bin_a(a))

    a = 17.3 # 8.5
    print  'a %4.1f  -->   %4d' % (a, bin_a(a))


    t = -180+17.1 # 8.5
    print  'a %4.1f  -->   %4d' % (t, bin_t(t))

    t = -180+17.3 # 8.5
    print  'a %4.1f  -->   %4d' % (t, bin_t(t))

    # for a in range(0,361,30):
    #     clsInd = bin_a(a)
    #     print  'a %4d  -->   %4d' % (a, clsInd)
    #
    # print '---------------------------------'
    # for t in range(-180,180+1, 30):
    #     clsInd = bin_t(t)
    #     print  't %4d  -->   %4d' % (t, clsInd)

#test0()
#exit()
# ------------------------------------------------
#   a    0  -->      0         t -180  -->     20
#   a   30  -->      1         t -150  -->      0
#   a   60  -->      3         t -120  -->      2
#   a   90  -->      5         t  -90  -->      4
#   a  120  -->      7         t  -60  -->      6
#   a  150  -->      8         t  -30  -->      7
#   a  180  -->     10         t    0  -->      9
#   a  210  -->     12         t   30  -->     11
#   a  240  -->     14         t   60  -->     13
#   a  270  -->     15         t   90  -->     14
#   a  300  -->     17         t  120  -->     16
#   a  330  -->     19         t  150  -->     18
#   a  360  -->     21         t  180  -->     20
# -------------------------------------------------



def get_intervals(nr_bin):
    step  = 360./nr_bin
    start = 360./(nr_bin*2)
    stop  = 360.-(360./(nr_bin*2)) +1e-2 # +1e-2 just because np.arange will exclude right boundary.
    return np.arange(start, stop, step)

#print get_intervals(24),
#print len(get_intervals(24))
# [0 (360/(b*2)):(360/b):360-(360/(b*2))]

def get_binInd(a, nr_bin=24):
    '''  # inter0     inter1                  interN-1#
         |    |         |              |         |    |
     <---|bin0|   bin1  |              |  binN-1 |bin0|--->
    |----#----|---------|--  ......  --|---------|----#----|
         0===========================================360
    '''
    bin_size = 360./ nr_bin
    a = ( a-(bin_size/2.) ) % 360   #  shift half bin left
    scaled_a = a / bin_size + 1
    return np.int32( np.floor(scaled_a) ) % nr_bin

# print get_binInd(8)
# print get_binInd(7.6)
# print get_binInd(352.4)
# print get_binInd(352.5)


class bin_quantizer:
    '''  # inter0     inter1                  interN-1#
         |    |         |              |         |    |
      <--|bin0|   bin1  |              |  binN-1 |bin0|-->
      ---#----|---------|--  ......  --|---------|----#---
         0===========================================360

    ____________[Logical Code]_____________
      bin_size = 360./ nr_bin
      a = ( a-(bin_size/2.) ) % 360   #  shift left by half bin size
      scaled_a = a / bin_size + 1     #  get scaled a w.r.t bin size.
      return np.int32( scaled_a ) % nr_bin
      # return np.int32( np.floor(scaled_a) ) % nr_bin
    '''
    def __init__(self, nr_bin):
        self.nr_bin = nr_bin
        self.bin_size = 360./self.nr_bin
        # Get intervals and bin certers
        _step  = self.bin_size
        _start = self.bin_size/2.  # start from half bin size
        _stop  = 360.-(self.bin_size/2.) +1e-2 # +1e-2 just because np.arange will exclude right boundary.
        self.intervals = np.arange(_start, _stop, _step)  # [start, stop)
        self.bincenter = np.arange(0, 360, _step)
        assert len(self.intervals)==self.nr_bin
        assert len(self.bincenter)==self.nr_bin

    def __call__(self, a):
        """input a ([0,360]), return binned index (int32)"""
        return  np.int32( (( a-(self.bin_size/2.) ) % 360) / self.bin_size + 1) % self.nr_bin

    # @property
    # def intervals(self):
    #     step  = self.bin_size
    #     start = self.bin_size/2.  # start from half bin size
    #     stop  = 360.-(self.bin_size/2.) +1e-2 # +1e-2 just because np.arange will exclude right boundary.
    #     return np.arange(start, stop, step)

def test1():
    vbin = bin_quantizer(4)
    print vbin.intervals, vbin.bin_size
    print vbin(44), vbin(46), vbin(134), vbin(314), vbin(315), vbin(316)
    print vbin(-45), vbin(-46), vbin(-44)

    # print vbin(8)
    # print vbin(7.6)
    # print vbin(352.4)
    # print vbin(352.5)

vbin = bin_quantizer(24)
print vbin.intervals
print vbin.bincenter
# intervals = [  0,  7.5,  22.5,  37.5,  52.5,  67.5,  82.5,  97.5,  112.5,  127.5,  142.5,  157.5,  172.5,
#               187.5,  202.5,  217.5,  232.5,  247.5,  262.5,  277.5,  292.5,  307.5, 322.5,  337.5,  352.5]
# print len(sp)
