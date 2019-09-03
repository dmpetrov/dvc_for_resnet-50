from __future__ import print_function, division, absolute_import
import argparse
import torch
import os
import sys
import pretrainedmodels
import pretrainedmodels.utils as utils
from PIL import Image
from model import resnet50

sys.path.append('.')

# model_names = sorted(name for name in pretrainedmodels.__dict__
#                      if not name.startswith("__")
#                      and name.islower()
#                      and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='Resnet50')
parser.add_argument('--data_dir', type=str,
                    default='data')
arch = 'resnet50'


def main():
    global args
    args = parser.parse_args()

    # model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    model = resnet50(num_classes=1000, pretrained='imagenet')
    model.eval()

    # Load and Transform one input image
    tf_img = utils.TransformImage(model)
    img = os.path.join(args.data_dir, 'croco.jpg')
    # input_data = utils.LoadImage(img)
    input_data = Image.open(img)  # 3x400x225
    input_data = tf_img(input_data)  # 3x299x299
    input_data = input_data.unsqueeze(0)  # 1x3x299x299
    input = torch.autograd.Variable(input_data)

    # Load Imagenet Synsets
    with open(os.path.join(args.data_dir, 'imagenet_synsets.txt'), 'r') as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]: ' '.join(spl[1:]) for spl in splits}

    with open(os.path.join(args.data_dir, 'imagenet_classes.txt'), 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Make predictions
    output = model(input)  # size(1, 1000)
    max, argmax = output.data.squeeze().max(0)
    class_id = argmax.item()
    class_key = class_id_to_key[class_id]
    classname = key_to_classname[class_key]

    print("'{}': '{}' is a '{}' with {}% confidence".format(arch, img, classname, round(max.item(), 2)))


if __name__ == '__main__':
    main()
