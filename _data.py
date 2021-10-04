import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset
import torchio as tio
from torchio.transforms.augmentation.spatial.random_affine import TypeOneToSixFloat


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
        target = torch.from_numpy(np.asarray(nii_target.dataobj)).type(torch.float)

        # clamp the data to a range of -1000 to 1000, provided of the authors of the model genesis
        data = torch.clamp(data, -1000, 1000)

        # normalize the data between 0 and 1
        if self.transform:
            data, target = self.transform((data, target))

        target = F.one_hot(target.long(), 3).permute(3, 0, 1, 2).float()
        return data.unsqueeze(0), target, img_name


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
        img, target = sample
        d, h, w = self.output_size
        img = F.interpolate(img.unsqueeze(0).unsqueeze(1), size=(d, h, w), mode='trilinear')
        target = F.interpolate(target.unsqueeze(0).unsqueeze(1), size=(d, h, w))
        return img.squeeze(0).squeeze(0), target.squeeze(0).squeeze(0)


class torchIO_ZNormalization(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        img, target = sample
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img.unsqueeze(0)),
            mask=tio.LabelMap(tensor=target.unsqueeze(0)))
        z_normalization = tio.ZNormalization()
        subj = z_normalization(subject)
        return subj.image.data.squeeze(0), subj.mask.data.squeeze(0)


class torchIO_RandomNoise(object):

    def __init__(self, std=(0, 0.25), mean=0):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        img, target = sample
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img.unsqueeze(0)),
            mask=tio.LabelMap(tensor=target.unsqueeze(0)))
        random_noise = tio.RandomNoise(std=self.std, mean=self.mean)
        subj = random_noise(subject)
        return subj.image.data.squeeze(0), subj.mask.data.squeeze(0)


class torchIO_RandomFlip(object):

    def __init__(self, axes=0, flip_probability: float = 0.5):
        self.axes = axes
        self.flip_probability = flip_probability

    def __call__(self, sample):
        img, target = sample
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img.unsqueeze(0)),
            mask=tio.LabelMap(tensor=target.unsqueeze(0)))
        random_flip = tio.RandomFlip(axes=self.axes, flip_probability=self.flip_probability)
        subj = random_flip(subject)
        return subj.image.data.squeeze(0), subj.mask.data.squeeze(0)


class torchIO_RandomAffine(object):

    def __init__(self, scales: TypeOneToSixFloat = 0.1,
                 degrees: TypeOneToSixFloat = 10,
                 translation: TypeOneToSixFloat = 0,
                 isotropic: bool = False,
                 center: str = 'image',
                 default_pad_value='minimum',
                 image_interpolation: str = 'linear', ):
        self.scales = scales
        self.degrees = degrees
        self.translation = translation
        self.isotropic = isotropic
        self.center = center
        self.default_pad_value = default_pad_value
        self.image_interpolation = image_interpolation

    def __call__(self, sample):
        img, target = sample
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img.unsqueeze(0)),
            mask=tio.LabelMap(tensor=target.unsqueeze(0)))
        affine = tio.RandomAffine(scales=self.scales,
                                  degrees=self.degrees,
                                  translation=self.translation,
                                  isotropic=self.isotropic,
                                  center=self.center,
                                  default_pad_value=self.default_pad_value,
                                  image_interpolation=self.image_interpolation)
        subj = affine(subject)
        return subj.image.data.squeeze(0), subj.mask.data.squeeze(0)


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
        img, _ = sample
        minn, maxx = self.norm_range

        return (img - minn) / (maxx - minn), _
