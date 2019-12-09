# CS461 Final Project:

## Repository Organization
    cs461-final-project/
      |- README.md
      |- post_event_selection.py --> Select images that correspond with specific code to post-process
      |- retrieve_ground_truth.py --> Parse GeoJSON crowd-sourced damage labels and save associated image tiles
      |- scrape_digitalglobe.py --> Scrape links to Pre-/Post-Event satellite images, download and compress TIF images
      |- split_data.py --> Perform 60-20-20 train,test,validate split on roads training dataset
      |- calculate_per_pixel_change.py --> Binarize image mask and compute Manhattan and Zero norms
      |- data_preprocessing.py --> Apply density filter and image augmentation to training dataset
      |- image_matching.py --> Perform SURF feature matching and perspective transformation for pre and post image matching
      |- prepare_satellite_imgs.py --> Apply 512x512 image tiling, (optionally) add contrast to images, and match pre-/post-images
      |- notebooks/
         |- UNet.ipynb
         |- damage_ground_truth.ipynb
      |- data/
         |- post_event
         	|- ...
         |- pre_event
         	|- ...


## Intructions for Running Satellite Image Road Segmentation Example

## Data Preprocessing
The directory containing the data to process is used as an argument when calling the program. Here is an example command to apply preprocessing to the `train` images in the `roads` directory.

```python data_preprocessing.py data/roads/train/```


Install requirements: pip3 install requirements.txt
