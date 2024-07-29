import timm
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
import os

from data.medical import NCT_CRC_HE
from utils.logging import Logger

def test(model, test_loader, epoch, best_top1):
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
            #     print("  top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
            #     print("  top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))
    
    print(f"Total accuracy (Epoch {epoch})")
    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))
    print()

    if (correct_top1 / total) > best_top1:
        best_top1 = correct_top1

    return best_top1


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # train configs
    batch_size = 32
    epochs = 150
    num_workers = 16

    # setup dataset and dataloader
    root_data_path = "/home/jovyan/dataset/NCT-CRC-HE-100K"
    train_set = NCT_CRC_HE(root_data_path, split="train")
    test_set = NCT_CRC_HE(root_data_path, split="val")
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

    # load the model
    model = timm.models.vit_base_patch16_224(pretrained=True, num_classes=train_set.num_classes).to(device)
    # model = timm.models.vit_large_patch16_224(pretrained=True, num_classes=train_set.num_classes).to(device)
    print("The number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # setup the optimizer
    lr = 0.0001
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # create logger
    logger = Logger("vit-b-nct", isStart=True)
    
    # main loop
    train_size = len(train_set)
    best_top1 = 0.0
    prev_best_top1 = 0.0
    for epoch in range(epochs):
        model.train()
        for batch, (images, labels) in tqdm(enumerate(train_loader), desc=f"Train {epoch} epoch: ", total=len(train_loader), leave=True):
            # get images and labels
            images = images.to(device)  # [100, 3, 224, 224]
            labels = labels.to(device)  # [100]

            # get outputs
            outputs = model(images)

            # optimize
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if batch % 3000 == 0:
            #     loss, current = loss.item(), batch * batch_size + len(images)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")

        # show loss value in current epoch
        loss, current = loss.item(), batch * batch_size + len(images)
        print(f"Epoch {epoch}, Loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")
            
        # validation    
        best_top1 = test(model, test_loader, epoch, best_top1)
        if best_top1 > prev_best_top1:
            prev_best_top1 = best_top1
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "ckpt", f"best_top1_{best_top1:.3f}.pt"))
