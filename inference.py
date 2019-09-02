from __future__ import print_function, division, absolute_import
import argparse
import torch
import os
import sys
import pretrainedmodels
import pretrainedmodels.utils as utils
sys.path.append('.')


model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='Resnet50 + faster rcnn')
parser.add_argument('--data_dir', type=str,
                    default='/home/sanjar/Desktop/xperience/dvc_project/dvc-in-use-example/data')
arch = 'resnet50'


def main():
    global args
    args = parser.parse_args()

    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    model.eval()

    # Load and Transform one input image
    load_img = utils.LoadImage()
    tf_img = utils.TransformImage(model)

    input_data = load_img(os.path.join(args.data_dir, 'croco.jpg'))  # 3x400x225
    input_data = tf_img(input_data)  # 3x299x299
    input_data = input_data.unsqueeze(0)  # 1x3x299x299
    input = torch.autograd.Variable(input_data)

    # Load Imagenet Synsets
    with open(os.path.join(args.data_dir, 'imagenet_synsets.txt'), 'r') as f:
        synsets = f.readlines()

    # len(synsets)==1001
    # sysnets[0] == background
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

    print()
    print("'{}': '{}' is a '{}'".format(arch, 'croco.jpg', classname))


if __name__ == '__main__':
    main()
