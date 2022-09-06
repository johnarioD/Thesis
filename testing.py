import pandas as pd
from matplotlib import pyplot as plt
import cv2

def new_dataset():
    metadata = pd.read_csv("./data/unsorted/metadata.csv")
    images = metadata.img_id[metadata.diagnostic == "BCC"]
    for image in images:
        res = plt.imread("./data/unsorted/images/"+image+".png")
        plt.imsave("./data/unprocessed/train/"+image+".jpg", res)


def repro():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    data = torch.randn([32, 64, 32, 32], dtype=torch.half, device='cuda', requires_grad=True)
    net = torch.nn.Conv2d(64, 128, kernel_size=[1, 1], padding=[0, 0], stride=[2, 2], dilation=[1, 1], groups=1)
    net = net.cuda().half()
    out = net(data)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()


if __name__ == "__main__":
    repro()