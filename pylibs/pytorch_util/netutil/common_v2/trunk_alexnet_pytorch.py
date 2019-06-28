import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from basic.common import rdict
import numpy as np
from easydict import EasyDict as edict

from pytorch_util.libtrain import copy_weights
from pytorch_util.torch_3rd_layers import Maskout
from pytorch_util.torch_3rd_funcs  import norm2unit

__all__ = ['AlexNet_Trunk'] # , 'alexnet']


# nr_cate = 3
'''                                                 Name_in_caffe
features.0.weight     -->  features.0.weight            conv1
features.0.bias       -->  features.0.bias
features.3.weight     -->  features.3.weight            conv2
features.3.bias       -->  features.3.bias
features.6.weight     -->  features.6.weight            conv3
features.6.bias       -->  features.6.bias
features.8.weight     -->  features.8.weight            conv4
features.8.bias       -->  features.8.bias
features.10.weight    -->  features.10.weight           conv5
features.10.bias      -->  features.10.bias
classifier.1.weight   -->  classifier.1.weight          fc6
classifier.1.bias     -->  classifier.1.bias
classifier.4.weight   -->  classifier.4.weight          fc7
classifier.4.bias     -->  classifier.4.bias
classifier.6.weight   -->  classifier.6.weight          fc8
classifier.6.bias     -->  classifier.6.bias
'''

class AlexNet_Trunk(nn.Module):
    def __init__(self):
        super(AlexNet_Trunk, self).__init__()

        #-- Trunk architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  #[ 0] Conv1   Note: bvlc alexnet is 96 instead of 64 here.
            nn.ReLU(inplace=True),                                  #[ 1]
            nn.MaxPool2d(kernel_size=3, stride=2),                  #[ 2]
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           #[ 3] Conv2
            nn.ReLU(inplace=True),                                  #[ 4]
            nn.MaxPool2d(kernel_size=3, stride=2),                  #[ 5]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          #[ 6] Conv3
            nn.ReLU(inplace=True),                                  #[ 7]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          #[ 8] Conv4
            nn.ReLU(inplace=True),                                  #[ 9]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          #[10] Conv5
            nn.ReLU(inplace=True),                                  #[11]
            nn.MaxPool2d(kernel_size=3, stride=2),                  #[12]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),                                           #[0]          Note: bvlc alexnet Dropout should follow after Fc
            nn.Linear(256 * 6 * 6, 4096),                           #[1] Fc6
            nn.ReLU(inplace=True),                                  #[2]
            nn.Dropout(),                                           #[3]
            nn.Linear(4096, 4096),                                  #[4] Fc7
            nn.ReLU(inplace=True),                                  #[5]
            # nn.Linear(4096, 1000), # num_classes),                #[6] Prob     Removed.
        )

    #@Interface
    def forward_Conv(self, x):
        return self.features(x)    # up to pool5

    #@Interface
    def forward_Fc  (self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)  # up to fc7

    # def forward_trunk(self, x):
    #     """ x is input image data.
    #         x is of shape:  (batchsize, 3, 227, 227)  """
    #     # forward convolutional layers
    #     x = self.features(x)
    #     #
    #     # forward fully connected layers
    #     batchsize = x.size(0)   # .split(1, dim=1)
    #     x = x.view(batchsize, 256 * 6 * 6)
    #     x = self.classifier(x)
    #     #
    #     return x

    #@Interface
    def forward(self, x): # In-replace forward_trunk
        """ x is input image data.
            x is of shape:  (batchsize, 3, 227, 227)  """
        # forward convolutional layers
        x = self.features(x)
        #
        # forward fully connected layers
        batchsize = x.size(0)   # .split(1, dim=1)
        x = x.view(batchsize, 256 * 6 * 6)
        x = self.classifier(x)
        #
        return x

    @staticmethod
    def init_weights(self):
        copy_weights(self.state_dict(), 'torchmodel.alexnet', strict=False)
        return self





class Test_AlexNet(nn.Module):

    def __init__(self, nr_cate=3,  _Trunk=AlexNet_Trunk):
        super(Test_AlexNet, self).__init__()
        self.truck = _Trunk().copy_weights()

        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)


    def forward(self, x, label):
        x = self.truck(x)
        # your code here
        batchsize = x.size(0)

        #-- Head architecture
        self.head_s2 = nn.Sequential(
            nn.Linear(4096, 84),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(84,  self.nr_cate*3),   # 252=3*3
            )
        self.head_s1 = nn.Sequential(
            nn.Linear(4096, 84),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(84,  self.nr_cate*2),
            )
        # self.maskout = Maskout(nr_cate=nr_cate)

        # Note: s1(x,y) is on a circle and (x^2+y^2=1)
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3) , label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2) , label)
        #-- Normalize coordinate to a unit
        x_s2 = norm2unit(x_s2) #, nr_cate=self.nr_cate)
        x_s1 = norm2unit(x_s1) #, nr_cate=self.nr_cate)

        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


if __name__ == '__main__':
    model = Test_AlexNet()  #.copy_weights() # pretrained=True

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,227,227),dtype=np.float32)
    dummy_batch_label = np.zeros((2,1),        dtype=np.int64)
    dummy_batch_data  = torch.autograd.Variable( torch.from_numpy(dummy_batch_data ) )
    dummy_batch_label = torch.autograd.Variable( torch.from_numpy(dummy_batch_label) )

    Pred = model(dummy_batch_data, dummy_batch_label)
    print (Pred.s2)



