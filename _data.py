import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset


class Nibabel_Dataset(Dataset):
    def __init__(self, path_list, labels_root_dir, training_root_dir, transform=None):
        # load all nii handle in a list
        self.labels_root_dir = labels_root_dir
        self.training_root_dir = training_root_dir
        self.train_list = [(nib.load(os.path.join(self.training_root_dir, image_path)), image_path) for (image_path, _)
                           in path_list]
        self.target_list = [(nib.load(os.path.join(self.labels_root_dir, label_path)), label_path) for (_, label_path)
                            in path_list]
        self.transform = transform

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        nii_image, img_name = self.train_list[idx]
        nii_target, label_name = self.target_list[idx]
        data = torch.from_numpy(np.asarray(nii_image.dataobj))

        # clamp the data to a range of -1000 to 1000, provided of the authors of the model genesis
        data = torch.clamp(data, -1000, 1000)

        # normalize the data between 0 and 1
        if self.transform:
            data = self.transform(data)

        target = torch.from_numpy(np.asarray(nii_target.dataobj)).type(torch.float)
        target = F.interpolate(target.unsqueeze(0).unsqueeze(1), size=(128, 80, 128), mode="nearest").squeeze().long()

        target = F.one_hot(target, 3).permute(3, 0, 1, 2).float()
        return data.unsqueeze(0), target, img_name  # .unsqueeze(0)


def load_img(path):
    nib_img = nib.load(path).get_fdata()
    return torch.from_numpy(nib_img).unsqueeze(0).unsqueeze(1).float()


class Rescale(object):
    """Add a padding if the obejct shape is not the same.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        d, h, w = self.output_size
        img = F.interpolate(image.unsqueeze(0).unsqueeze(1), size=(d, h, w))
        return img.squeeze(0).squeeze(0)


class Normalize(object):
    """Add a padding if the obejct shape is not the same.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, r):
        assert isinstance(r, tuple)
        self.norm_range = r

    def __call__(self, sample):
        minn, maxx = self.norm_range

        return (sample - minn) / (maxx - minn)
