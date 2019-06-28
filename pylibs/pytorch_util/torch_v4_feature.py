import math
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter


from torch.nn.functional import * # pad, avg_pool2d


# this is already in v4.0, but now I'm using v3.0
def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1):
    """Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)

    return input / div

class LocalResponseNorm(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        r"""Applies local response normalization over an input signal composed
        of several input planes, where channels occupy the second dimension.
        Applies normalization across channels.

        .. math::

            `b_{c} = a_{c}\left(k + \frac{\alpha}{n}
            \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}`

        Args:
            size: amount of neighbouring channels used for normalization
            alpha: multiplicative factor. Default: 0.0001
            beta: exponent. Default: 0.75
            k: additive factor. Default: 1

        Shape:
            - Input: :math:`(N, C, ...)`
            - Output: :math:`(N, C, ...)` (same shape as input)
        Examples::
            >>> lrn = nn.LocalResponseNorm(2)
            >>> signal_2d = autograd.Variable(torch.randn(32, 5, 24, 24))
            >>> signal_4d = autograd.Variable(torch.randn(16, 5, 7, 7, 7, 7))
            >>> output_2d = lrn(signal_2d)
            >>> output_4d = lrn(signal_4d)
        """
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        # return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)
        return local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.size) \
            + ', alpha=' + str(self.alpha) \
            + ', beta=' + str(self.beta) \
            + ', k=' + str(self.k) + ')'
