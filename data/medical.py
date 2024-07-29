import torch.nn as nn
import torchvision.transforms as transforms

import os
from glob import glob
from PIL import Image


class NCT_CRC_HE(nn.Module):
    def __init__(self, 
                 root_path,
                 split="val",
                 transform=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # get filenames
        if split == "val":
            self.dataset_path = os.path.join(root_path, "CRC-VAL-HE-7K")
        elif split == "train":
            self.dataset_path = os.path.join(root_path, "NCT-CRC-HE-100K")
        else:
            raise Exception("Specify the 'split' parameter")
        
        self.image_files_path, self.labels, self.length, self.num_classes, self.class_names = self.read_data()

        # transforms
        if transform is None:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.transforms = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            self.transforms = transform

    def read_data(self):
        all_img_fnames = list()
        all_labels = list()

        class_names = os.walk(self.dataset_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.dataset_path, class_name)
            img_fnames = os.walk(img_dir).__next__()[2]

            for img_f in img_fnames:
                img_fname = os.path.join(img_dir, img_f)
                # img = Image.open(img_file)
                if os.path.exists(img_fname) is not None:
                    all_img_fnames.append(img_fname)
                    all_labels.append(label)

        return all_img_fnames, all_labels, len(all_img_fnames), len(class_names), class_names

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.labels[index]
    
    def __len__(self):
        return self.length
