import shutil
import pandas as pd

def main():
    ISIC_2019 = pd.read_csv("./data/unsorted/ISIC_2019_Training_GroudTruth.csv")
    PAD_UFES = pd.read_csv("./data/unsorted/PAD_UFES.csv")
    for image in ISIC_2019['image'][ISIC_2019['BCC']==1]:
        shutil.copy("./data/unsorted/ISIC_2019_Training_Input/"+image,"./data/unprocessed/SSL_data/"+image)
    for image in PAD_UFES['img_id'][PAD_UFES['diagnostic']=="BCC"]:
        shutil.copy("./data/unsorted/PAD_UFES/"+image,"./data/unprocessed/SSL_data/"+image)

if __name__ == "__main__":
    main()
    