import os
import torch
import argparse
import torchvision.models as models

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--ckpt_path', type=str, required=True,
                    help='pytorch lightning checkpoint path')

parser.add_argument('--save_path', type=str, required=True,
                    help='path to save pytorch checkpoint')

parser.add_argument('--arch', type=str, required=True,
                    help='model arch')

args = parser.parse_args()

arch_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101
}


def convert_model(ckpt_path, save_path, arch):

    net = arch(pretrained=False)
    moco_ckpt = torch.load(ckpt_path)

    state_dict = moco_ckpt['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith(
                'module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    net.load_state_dict(state_dict, strict=False)

    torch.save(net.state_dict(), save_path)


convert_model(args.ckpt_path, args.save_path, arch_dict[args.arch])
