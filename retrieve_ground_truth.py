import argparse
import pandas as pd
import geopandas as gpd
import os
import numpy as np
from collections import namedtuple
from PIL import Image
import requests
from io import BytesIO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tomnod_file", help="Image directory")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    df = gpd.read_file(os.path.join(os.pardir, os.pardir, 'data', args.tomnod_file))
    Label = namedtuple('Label', 'link id')
    df = df.loc[(df.label == 'Flooded / Blocked Road') | (df.label == 'Blocked Bridge')]
    labels = [Label(df.iloc[i].chip_url, df.iloc[i].tag_id)  for i in range(len(df.index))]

    for label in labels:
        response = requests.get(label.link)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(os.pardir, 'data/ground_truth', str(label.id)) + '.png')

if __name__ == "__main__":
    main()
