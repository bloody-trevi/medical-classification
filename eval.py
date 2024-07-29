import argparse

import timm
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm

from data.medical import NCT_CRC_HE


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", dest="ckpt", action="store")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    root_data_path = "/home/jovyan/dataset/NCT-CRC-HE-100K"
    test_set = NCT_CRC_HE(root_data_path, split="val")
    test_loader = data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

    model = timm.models.vit_base_patch16_224(pretrained=True, num_classes=test_set.num_classes).to(device)
    # model = timm.models.vit_large_patch16_224(pretrained=True).to(device)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))
    
    print("The number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch, (images, labels) in tqdm(enumerate(test_loader), desc="Test: ", total=len(test_loader)):

            images = images.to(device)  # [100, 3, 224, 224]
            labels = labels.to(device)  # [100]
            outputs = model(images)

            # rank 1
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            # rank 5
            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

            # main loop
            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_k.item()

            # if batch % 100 == 0:
            #     print("step : {} / {}".format(idx + 1, len(test_set) / int(labels.size(0))))
            #     print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
            #     print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))

    print("Total accuracy")
    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))
    print(total)