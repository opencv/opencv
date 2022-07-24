import sys
from typing import List, Union
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import torchvision.transforms as T
import PIL
import torch


_available_datasets = {
    "cifar-10":  "/Users/chuyang/Downloads/cifar-10-batches-py",
    "imagenet-full": "",
    "imagenet-320": "/Users/chuyang/Downloads/imagenette2-320/train/n01440764",
}


def load_dataset(dataset: str, return_tensor: bool = False) -> List[Union[np.array, torch.Tensor]]:
    """
    Args:
    - dataset: (str) the name of the dataset
    - return_tensor: (bool) return tensor or not. return np.ndarray if set to False, otherwise return torch.tensor

    Returns:
        list of loaded image files
    """
    assert dataset in _available_datasets

    if dataset == 'cifar-10':
        path = _available_datasets[dataset]
        data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        # use data_batch_1 by default
        with open(os.path.join(path, data_files[0]), 'rb') as fb:
            raw_data = pickle.load(fb, encoding='bytes')
        data = raw_data[b'data']

        all_images = []
        data_wrapper = lambda x: torch.tensor(x).permute(2, 0, 1) if return_tensor else x
        for i in range(len(data)):
            img = data[i, :].reshape((32, 32, 3), order='F')
            all_images.append(data_wrapper(img))

        return all_images

    if dataset in ["imagenet-full", "imagenet-320"]:
        path = _available_datasets[dataset]
        all_images = []
        data_wrapper = lambda x: torch.tensor(x).permute(2, 0, 1) if return_tensor else x
        for filename in os.listdir(path):
            if filename.split('.')[-1] == 'JPEG':
                img = cv2.imread(os.path.join(path, filename))
                all_images.append(data_wrapper(img))
        return all_images

    raise NotImplementedError("dataset {} is not recognized!".format(dataset))


def runtime(func):
    def func_wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - start
    return func_wrapper


@runtime
def ocv_aug_func(dataset, transform, arguments):
    for img in dataset:
        transform(img, *arguments)


@runtime
def ocv_aug_class(dataset, transform):
    for img in dataset:
        transform.call(img)


@runtime
def pytorch_aug_class(dataset, transform):
    for img in dataset:
        transform(img)


# def test_ocv_random_crop_func():
#     cifar_10 = load_dataset("cifar-10")
#     transform = cv2.randomCrop
#     arguments = [np.array([20, 20])]
#     total_time = ocv_aug_func(cifar_10, transform, arguments)
#     print("test_ocv_random_crop_func", total_time)
#     del cifar_10


def test_ocv_random_crop_func():
    dataset = load_dataset("imagenet-320")
    transform = cv2.randomCrop
    # transform = cv2.randomCropV1
    arguments = [np.array([200, 200])]
    total_time = ocv_aug_func(dataset, transform, arguments)
    print("test_ocv_random_crop_func", total_time)
    del dataset


def test_ocv_resize_cls():
    dataset = load_dataset("imagenet-320")
    transform = cv2.Resize((300, 300))
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_pytorch_resize():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.Resize((200, 200))
    used_time = pytorch_aug_class(dataset, transform)
    return used_time


def test_pytorch_random_crop():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.RandomCrop((200, 200))
    total_time = pytorch_aug_class(dataset, transform)
    print("test_pytorch_random_crop", total_time)
    del dataset


def test_ocv_center_crop_cls():
    dataset = load_dataset("imagenet-320")
    transform = cv2.CenterCrop(np.array([200, 200]))
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_ocv_center_crop_func():
    dataset = load_dataset("imagenet-320")
    transform = cv2.centerCrop
    arguments = [np.array([200, 200])]
    used_time = ocv_aug_func(dataset, transform, arguments)
    return used_time


def test_pytorch_center_crop():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.CenterCrop((200, 200))
    total_time = pytorch_aug_class(dataset, transform)
    return total_time


def test_ocv_pad_cls():
    dataset = load_dataset("imagenet-320")
    transform = cv2.Pad(np.array([100, 100, 100, 100]))
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_pytorch_pad():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.Pad((100, 100, 100, 100))
    total_time = pytorch_aug_class(dataset, transform)
    return total_time


def test_ocv_random_crop_cls():
    dataset = load_dataset("imagenet-320")
    transform = cv2.RandomCrop(np.array([200, 200]))
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_pytorch_random_crop():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.RandomCrop((200, 200))
    total_time = pytorch_aug_class(dataset, transform)
    return total_time


def test_ocv_random_resized_crop():
    dataset = load_dataset("imagenet-320")
    transform = cv2.RandomResizedCrop(np.array([500, 500]))
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_pytorch_random_resized_crop():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.RandomResizedCrop((500, 500))
    total_time = pytorch_aug_class(dataset, transform)
    return total_time


def test_pytorch_compose_three():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.Compose([
        T.RandomCrop(300, 300),
        T.RandomHorizontalFlip(),
        T.Pad((100, 100, 100, 100))
    ])
    total_time = pytorch_aug_class(dataset, transform)
    return total_time


def test_ocv_compose_three():
    dataset = load_dataset("imagenet-320")
    t1 = cv2.RandomCrop(np.array([300, 300]))
    t2 = cv2.RandomFlip()
    t3 = cv2.Pad(np.array([100, 100, 100, 100]))
    transform = cv2.Compose([t1, t2, t3])
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_ocv_compose_four():
    dataset = load_dataset("imagenet-320")
    t1 = cv2.Resize(np.array([400, 400]))
    t2 = cv2.Pad(np.array([100, 100, 100, 100]))
    t3 = cv2.RandomFlip()
    t4 = cv2.CenterCrop(np.array([200, 200]))
    transform = cv2.Compose([t1, t2, t3, t4])
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_pytorch_compose_four():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.Compose([
        T.Resize([400, 400]),
        T.Pad((100, 100, 100, 100)),
        T.RandomHorizontalFlip(),
        T.CenterCrop((200, 200))
    ])
    total_time = pytorch_aug_class(dataset, transform)
    return total_time


def test_ocv_random_flip_cls():
    dataset = load_dataset("imagenet-320")
    transform = cv2.RandomFlip()
    used_time = ocv_aug_class(dataset, transform)
    return used_time


def test_pytorch_random_flip():
    dataset = load_dataset("imagenet-320", return_tensor=True)
    transform = T.RandomHorizontalFlip()
    total_time = pytorch_aug_class(dataset, transform)
    return total_time




def main():
    all_time = []
    for i in range(5):
        all_time.append(test_pytorch_compose_four())
    print(all_time)
    print(sum(all_time)/len(all_time))


if __name__ == '__main__':
    print(sys.argv)
    main()
