from typing import ClassVar
import paddle
import paddle.nn as nn

from models.Model import vgg19

class pre_network(nn.Layer):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self, pretrained: str = None):
        super(pre_network, self).__init__()
        self.vgg_layers = vgg19(pretrained=pretrained).features
        self.layer_name_mapping = {
            '3': 'relu1',
            '8': 'relu2',
            '13': 'relu3',
            # '22':'relu4',
            # '31':'relu5',
        }

    def forward(self, x):
        output = {}

        for name, module in self.vgg_layers._sub_layers.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output