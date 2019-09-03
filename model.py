# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import io
import torchvision.models as models
import dvc.api as dvc_api
import torch
import types

__all__ = ['resnet50']

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }


def load_pretrained(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    # with dvc_api.open('model_weights/resnet50-19c8e357.pth', remote='gsremote', mode="rb", encoding=None) as weights:
    #     model_weights = torch.load(io.BytesIO(weights))
    # model.load_state_dict(model_weights)
    weights = dvc_api.read('model_weights/resnet50-19c8e357.pth', remote='gsremote', mode="rb", encoding=None)
    model_weights = torch.load(io.BytesIO(weights))
    model.load_state_dict(model_weights)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


# ResNets

def modify_resnets(model):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None
    model.softmax = torch.nn.Softmax(dim=1)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        x = self.softmax(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet50(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet50(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model
