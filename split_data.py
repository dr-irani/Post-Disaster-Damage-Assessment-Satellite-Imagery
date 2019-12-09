import argparse
import os
import pandas as pd
import numpy as np


# def get_args():
#     parser = argparse.ArgumentParser(description="Scrape GeoTIFF images from DigitalGlobe.")
#     parser.add_argument('event_url', metavar='link to event on DigitalGlobe', type=str)
#     parser.add_argument('disaster_name', help='specify name of disaster to create output folder')
#     parser.add_argument('--directory', metavar='testing file', default=, type=str)
#     args = parser.parse_args()
#     return args


def get_data_split(directory = '/Volumes/ExtremeSSD/cs461_final_project/data/roads/train'):
    files = set([(f.split('.')[0]).split('_')[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    df = pd.DataFrame(files)
    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    return train, validate, test

def move_to_dir(train, test, validate, directory = '/Volumes/ExtremeSSD/cs461_final_project/data/roads/train'):
    os.chdir(directory)
    for t in train.values.tolist():
        os.system('cp ' + t[0] + '_sat.jpg train/')
        os.system('cp ' + t[0] + '_mask.png train/')

    # os.chdir('../test/')
    for t in test.values.tolist():
        os.system('cp ' + t[0] + '_sat.jpg test/')
        os.system('cp ' + t[0] + '_mask.png test/')

    # os.chdir('../validate/')
    for t in validate.values.tolist():
        os.system('cp ' + t[0] + '_sat.jpg validate/')
        os.system('cp ' + t[0] + '_mask.png validate/')


def main():
    # args = get_args()
    train, validate, test = get_data_split()
    move_to_dir(train, validate, test)
    


    
    


if __name__ == "__main__":
    main()