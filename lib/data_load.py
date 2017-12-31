import os
import torch
import torch.utils.data as data
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fh = open(label)
        imgs = []
        classes = []
        for line in fh.readlines():
            cls = line.split()
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn)) and len(cls) == 1:
                label = int(cls[0])
                imgs.append((fn, label))
                # imgs.append((fn, tuple([float(v) for v in cls])))
                classes.append(label)
        self.classes = sorted(list(set(classes)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)



def testmyImageFloder():
    dataloader = myImageFloder('/home/zkyang/Workspace/task/Pytorch_task/test_m/data/car',
                               '/home/zkyang/Workspace/task/Pytorch_task/test_m/data/train.txt')
    # print ('dataloader.getName', dataloader.getName())

    for index, (img, label) in enumerate(dataloader):
        # img.show()
        print (label)
    print (dataloader.classes)


if __name__ == "__main__":
    testmyImageFloder()