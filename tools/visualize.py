import os
import sys
from glob import glob
import argparse
import random

import torch
import torch.utils.data as data
import timm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.medical import NCT_CRC_HE


def visualize(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset
    root_data_path = "/home/jovyan/dataset/NCT-CRC-HE-100K"
    test_set = NCT_CRC_HE(root_data_path, split="val")
    # test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    # load model
    model = timm.models.vit_base_patch16_224(pretrained=True, num_classes=test_set.num_classes).to(device)
    # model = timm.models.vit_large_patch16_224(pretrained=True).to(device)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))
    
    print("The number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.eval()

    # get some predictions randomly (or specifically)
    pred_list = list()      # predicted class indices
    label_list = list()     # GT class indices
    input_list = list()     # input images
    num_samples = args.num_samples
    sample_idx = random.sample(range(len(test_set)), num_samples)
    with torch.no_grad():
        for i in sample_idx:
            img, label = test_set[i]
            outputs = model(img.to(device).unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            pred_list.append(pred.item())
            label_list.append(label)

            # convert images [-1,1] to [0,255]
            img_n = img.cpu().numpy().transpose(1,2,0)
            img_n[:,:,0] = img_n[:,:,0] * 0.5 + 0.5
            img_n[:,:,1] = img_n[:,:,1] * 0.5 + 0.5
            img_n[:,:,2] = img_n[:,:,2] * 0.5 + 0.5
            input_list.append(img_n)

    # match class names
    # cls_names = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    cls_names = test_set.class_names
    cls_list = [cls_names[i] for i in label_list]
    pred_cls_list = [cls_names[i] for i in pred_list]

    # visualize samples
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Medical Classification Results", fontsize=18, y=0.95)
    for n in range(len(pred_cls_list)):
        # add a new subplot iteratively
        ax = plt.subplot(3, 3, n + 1)
        ax.imshow(input_list[n])

        # chart formatting
        ax.set_title(f"GT: {cls_list[n]} / Pred: {pred_cls_list[n]}")
        ax.axis("off")

    # save img
    save_root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    saved_imgs_path = os.listdir(save_root_path)
    saved_imgs_path = [file for file in saved_imgs_path if file.endswith(".png")]
    
    plt.savefig("eval_result.png")


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", dest="ckpt", action="store", default="ckpt/best_top1_0.955.pt")
    parser.add_argument("-ns", "--num_samples", dest="num_samples", action="store", default=9)
    args = parser.parse_args()

    visualize(args)
