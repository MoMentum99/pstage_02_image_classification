import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url


class MaskBaseModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, freeze=False):
        super().__init__()

        self.build_layers(num_classes, pretrained)

        if not pretrained:
            self._initialize_weights()

        if freeze:
            self._freeze_net()

    def build_layers(self, num_classes, pretrained):
        raise NotImplementedError

    def backbone_layers(self):
        """ returns backbone layers to be referenced for weight freezing or lr targeting """

        raise NotImplementedError

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or lr targeting """

        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _freeze_net(self):
        backbone_layers = self.backbone_layers()
        for param in backbone_layers.parameters():
            param.requires_grad = False


class AlexNet(MaskBaseModel):

    def build_layers(self, num_classes, pretrained):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def backbone_layers(self):
        """ returns backbone layers to be referenced for weight freezing or lr targeting """

        return self.features

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or lr targeting """

        return self.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG19(MaskBaseModel):

    def build_layers(self, num_classes, pretrained):
        from torchvision.models import vgg19
        self.net = vgg19(pretrained=pretrained)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def backbone_layers(self):
        """ returns backbone layers to be referenced for weight freezing or lr targeting """

        return self.net.features

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or lr targeting """

        return self.net.classifier

    def forward(self, x):
        return self.net(x)
