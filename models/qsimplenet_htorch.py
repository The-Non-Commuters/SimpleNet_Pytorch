"""
SimplerNetV1 in Pytorch.

The implementation is basded on : 
https://github.com/D-X-Y/ResNeXt-DenseNet
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from htorch.layers import QConv2d, QBatchNorm2d, QMaxPool2d, QLinear, QuaternionToReal
from htorch.functions import QDropout, QDropout2d


class qsimplenet_htorch(nn.Module):
    def __init__(self, classes=10, N=16, simpnet_name="qsimplenet_htorch"):
        super().__init__()
        self.features = self._make_layers(N)
        self.classifier = nn.Sequential(
            QLinear(N*4, classes),
            QuaternionToReal(classes)
        )
        self.pool = lambda x, **kw: QMaxPool2d(**kw)(x)
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
            if isinstance(param, Parameter):
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
        out = self.pool(out, kernel_size=out.size()[2:])

        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, N, BN_cls=QBatchNorm2d):
        N_BN = N
        if not issubclass(BN_cls, QBatchNorm2d):
            N_BN *= 4

        model = nn.Sequential(
            QConv2d(1, N, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N, N*2, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*2, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N*2, N*2, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*2, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N*2, N*2, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*2, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(N*2, N*2, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*2, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N*2, N*2, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*2, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N*2, N*4, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*4, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(N*4, N*4, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*4, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N*4, N*4, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*4, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(N*4, N*8, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*8, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(N*8, N*16, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            BN_cls(N_BN*16, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QConv2d(N*16, N*4, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            BN_cls(N_BN*4, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            QMaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            QDropout2d(p=0.1),
            QConv2d(N*4, N*4, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            BN_cls(N_BN*4, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        )

        # for m in model.modules():
        #     if isinstance(m, QConv2d):
        #         nn.init.xavier_uniform_(
        #             m.weight.data, gain=nn.init.calculate_gain("relu")
        #         )

        return model
