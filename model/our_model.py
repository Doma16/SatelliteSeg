import torch
import torch.nn as nn
import torchinfo

from utils import read_json_variable, count_parameters

RESNET9_PATH = read_json_variable('paths.json', 'resnet9')

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):

    def __init__(self, in_channels, last_layer=False):
        super().__init__()

        POOL = False
        self.last_layer = last_layer

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=POOL)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=POOL)
        self.conv4 = conv_block(256, 512, pool=POOL)
        if last_layer:
            self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1),
        # )

        self.load_params(RESNET9_PATH)

    def load_params(self, path):
        state_dict = torch.load(path)
        del state_dict['classifier.3.weight']
        del state_dict['classifier.3.bias']
        if not self.last_layer:
            for state in [x for x in state_dict.keys() if 'res2.' in str(x)]:
                del state_dict[state]
        self.load_state_dict(state_dict)
        self.to('cpu')

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        if self.last_layer:
            out = self.res2(out) + out
        return out
    
class PatchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ResNet9(3)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, xb):
        out = self.backbone(xb)
        out = self.decoder(out)
        return out

class WholeBlock(nn.Module):
    def __init__(self, in_channels, last=False):
        super().__init__()
        DOWN = 8
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//DOWN, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//DOWN),
        )
        if not last:
            self.block.append(nn.ReLU())

    def forward(self, xb):
        return self.block(xb)

class WholeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'WholeModel'
        self.backbone = ResNet9(3)
        self.decoder = nn.Sequential(
            WholeBlock(512),
            WholeBlock(64),
            WholeBlock(8, last=True),
        )

    def forward(self, xb):
        out = self.backbone(xb)
        out = self.decoder(out)
        return out

if __name__ == '__main__':
    N = 2
    '''
    PatchModel
    '''
    model = PatchModel()
    torchinfo.summary(model, input_size=(1, 3, 32, 32), device='cpu')

    patches = torch.rand((N, 3, 16, 16)).to('cpu')
    out = model(patches)
    print(count_parameters(model))
    

    '''
    WholeModel
    '''
    model = WholeModel()
    torchinfo.summary(model, input_size=(1, 3, 32, 32), device='cpu')

    print(count_parameters(model.backbone))
    print(count_parameters(model.decoder))

    images = torch.randn((N, 3, 400, 400)).to('cpu')
    out = model(images)