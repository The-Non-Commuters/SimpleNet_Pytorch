"""
SimplerNetV1 in Pytorch.

The implementation is basded on : 
https://github.com/D-X-Y/ResNeXt-DenseNet
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from orkis.qnn import QConv2d, QLinear, QBatchNorm2d, QMaxPool2d, QDropout, QDropout2d, QuaternionToReal
from orkis.qnn.ops import qmax_pool2d

class qsimplenet_orkis(nn.Module):
    def __init__(self, classes=10, multiply_filters=True, simpnet_name="qsimplenet_orkis"):
        super().__init__()
        self.multiplier = 4 if multiply_filters else 1
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            QLinear(256*self.multiplier, classes*4),
            QuaternionToReal()
        )
        self.drp = QDropout(0.1)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        # print(own_state.keys())
        # for name, val in own_state:
        # print(name)
        for name, param in state_dict.items():
            name = name.replace("module.", "")
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print(
                    "While copying the parameter named {}, whose dimensions in the model are"
                    " {} and whose dimensions in the checkpoint are {}, ... Using Initial Params".format(
                        name, own_state[name].size(), param.size()
                    )
                )

    def forward(self, x):
        out = self.features(x)
        # Global Max Pooling
        out = qmax_pool2d(out, kernel_size=out.size()[2:])

        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):

        model = nn.Sequential(
            QConv2d(4, 64*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(64*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(64*self.multiplier, 128*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(128*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(128*self.multiplier, 128*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(128*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(128*self.multiplier, 128*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(128*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(128*self.multiplier, 128*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(128*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(128*self.multiplier, 128*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(128*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(128*self.multiplier, 256*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(256*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(256*self.multiplier, 256*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(256*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(256*self.multiplier, 256*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(256*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(256*self.multiplier, 512*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(512*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(512*self.multiplier, 2048*self.multiplier, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            QBatchNorm2d(2048*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(2048*self.multiplier, 256*self.multiplier, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            QBatchNorm2d(256*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(256*self.multiplier, 256*self.multiplier, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            QBatchNorm2d(256*self.multiplier, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        )

        # for m in model.modules():
        #     if isinstance(m, QConv2d):
        #         nn.init.xavier_uniform_(
        #             m.weight.data, gain=nn.init.calculate_gain("relu")
        #         )

        return model
