# flickr_csv_dataset_openclip.py

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FlickrOpenCLIPDataset(Dataset):
    def __init__(self, csv_path, preprocess):
        self.data = pd.read_csv(csv_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        image = self.preprocess(image)

        return {
            "image": image,
            "text": row["text"]  # 👈 Return raw text (this is NOW FIXED)
        }