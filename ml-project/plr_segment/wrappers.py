# coding=utf-8
from numpy import arange, array, ones
from numpy.linalg import lstsq
def leastsquareslinefit(sequence,seq_range):

    """Return the parameters and error for a least squares line fit of one segment of a sequence"""
    x = arange(seq_range[0],seq_range[1]+1)
    y = array(sequence[seq_range[0]:seq_range[1]+1])
    A = ones((len(x),2),float)
    A[:,0] = x
    #用最小二乘法拟合数据得到一个线性方程
    (p,residuals,rank,s) = lstsq(A,y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return (p,error)

if __name__ == '__main__':
    sequence=[1,4,7]