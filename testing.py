import pandas as pd
from matplotlib import pyplot as plt

metadata = pd.read_csv("./data/unsorted/ISIC_2019_Training_GroundTruth.csv")
images = metadata.image[metadata.BCC == 1]
for image in images:
    res = plt.imread("./data/unsorted/images/"+image+".jpg")
    plt.imsave("./data/unprocessed/train/"+image+".jpg", res)
