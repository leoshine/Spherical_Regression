"""
 @Author  : Shuai Liao
"""

# full assembly of the sub-parts to form the complete net
import numpy as np
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1  = self.inc(x)
        x2  = self.down1(x1)
        x3  = self.down2(x2)
        x4  = self.down3(x3)
        x5  = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        x_ = self.outc(x1_)
        print('x1 ', x1.shape)
        print('x2 ', x2.shape)
        print('x3 ', x3.shape)
        print('x4 ', x4.shape)
        print('x5 ', x5.shape)
        print('x4_', x4_.shape)
        print('x3_', x3_.shape)
        print('x2_', x2_.shape)
        print('x1_', x1_.shape)
        print('x_ ', x_ .shape)
        return x



if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=1)

    # import numpy as np
    dummy_batch_data  = np.zeros((2,3,224,224),dtype=np.float32)
    dummy_batch_data  = torch.from_numpy(dummy_batch_data )

    Pred = model(dummy_batch_data)
    print(Pred.shape)
