from __future__ import print_function, division, absolute_import
import argparse
import torch
import os
import sys
import pretrainedmodels.utils as utils
from PIL import Image
from model import resnet50

sys.path.append('.')


def run_eval(data_dir, model):
    model = model  # resnet50 model by default
    model.eval()

    # Load and Transform one input image
    tf_img = utils.TransformImage(model)
    img = os.path.join(data_dir, 'croco.jpg')
    input_data = Image.open(img)  # 3x400x225
    input_data = tf_img(input_data)  # 3x299x299
    input_data = input_data.unsqueeze(0)  # 1x3x299x299
    input = torch.autograd.Variable(input_data)

    # Load Imagenet Synsets
    with open(os.path.join(data_dir, 'imagenet_synsets.txt'), 'r') as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]: ' '.join(spl[1:]) for spl in splits}

    with open(os.path.join(data_dir, 'imagenet_classes.txt'), 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Make predictions
    output = model(input)  # size(1, 1000)
    max_, argmax = output.data.squeeze().max(0)
    class_id = argmax.item()
    class_key = class_id_to_key[class_id]
    classname = key_to_classname[class_key]
    return img, classname, max_, class_key


def main():
    arch = 'resnet50'
    parser = argparse.ArgumentParser(description='Resnet50')
    parser.add_argument('--data_dir', type=str, default='data', metavar='PATH', help='path to get data from repository')
    parser.add_argument('-m', '--model', type=object, default=resnet50(num_classes=1000, pretrained='imagenet'),
                        metavar='MODEL', help='model object that will be run on inference')
    args = parser.parse_args()
    data_dir = args.data_dir
    model = args.model


    img, classname, max_, class_key = run_eval(args)
    print("'{}': '{}' is a '{}' with {}% confidence".format(arch, img, classname, round(max_.item() * 100, 2)))


if __name__ == '__main__':
    main()

