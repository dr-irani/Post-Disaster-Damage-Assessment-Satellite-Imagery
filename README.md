# CS461 Final Project: Damage and Accessibility Assessment for Post-Disaster Regions from Satellite Imagery

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
         |- UNet.ipynb --> not pretrained unet model
         |- damage_ground_truth.ipynb --> Parse GeoJSON crowd-sourced damage labels and view/save associated image tiles
         |- data_preprocessing.ipynb --> preprocessing training data
         |- semantic_segmentation.ipynb --> unet/fpn model, change type by changing the model parameter
         |- segment_satellite.ipynb --> makes prediction masks for pre-/post-event satellite images
         |- per_pixel_overlap.ipynb --> calculates the per pixel overlaps from pre-/post-event mask
         |- best_model_full.pth --> best model of fpn saved over 40 iterations
         |- best_model_unet.pth --> best model of unet saved over 40 iterations
      |- data/
         |- post_event
         	|- ...
         |- pre_event
         	|- ...


## Intructions for Running Satellite Image Road Segmentation Example
1. Run through semantic_segmentation.ipynb. The current default is unet. If want to get the result for fpn, leave only ```model_path = './best_model_full.pth'``` uncommented.  
2. Run through per_pixel_overlap.ipynb. The current default is unet. If want to get result for fpn, uncomment ``` mask_dir = '../predictions_fpn/'```

## Data Preprocessing
The directory containing the data to process is used as an argument when calling the program. Here is an example command to apply preprocessing to the `train` images in the `roads` directory.

```python data_preprocessing.py data/roads/train/```

Install requirements: 
```pip3 install requirements.txt```
