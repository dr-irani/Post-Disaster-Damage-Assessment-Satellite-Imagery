import argparse
import os
import requests
from urllib.parse import urlparse
from urllib.request import urlretrieve
from collections import defaultdict
from bs4 import BeautifulSoup


def get_args():
    parser = argparse.ArgumentParser(description="Scrape GeoTIFF images from DigitalGlobe.")
    parser.add_argument('event_url', metavar='link to event on DigitalGlobe', type=str)
    args = parser.parse_args()
    return args


def no_return_error(links):
    if len(links) == 0:
        raise Exception('This DigitalGlobe website does not have any image download links')


def get_img_links(page_url):
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.findAll('textarea')

    no_return_error(links)

    return [link for chunk in links for link in chunk.text.split() if link.endswith('.tif')]

def get_img_by_date(links):
    pre_events = defaultdict(list)
    post_events = defaultdict(list)

    for link in links:
        url = urlparse(link)
        event = url.path.split('/')[2]
        date = url.path.split('/')[3]
        code = url.path.split('/')[4]
        if 'pre-event' in event and code == '103001004F0AF500': pre_events[date].append(link)
        elif 'post-event' in event and code == '10300100728F1700': post_events[date].append(link)

    return pre_events, post_events


def compress_tif(original_tif, compression_method="JPEG",
                 predictors=2, new_directory="compressed/"):
    """
    This function takes an uncompressed GeoTIFF and compresses it with one of
    four compression methods:

        - Packbits
        - JPEG
        - Deflate
        - LZW

    For LZW and Deflate, you can choose the number of predictors.

    :param original_tif: The uncompressed GeoTIFF to be compressed
    :param new_tif: The new compressed GeoTIFF
    :param compression_method: Packbits, JPEG, Deflate, or LZW
    :param predictors: Default is 2
    :return: Creates a new compressed TIF in directory folder
    """

    new_tif_base = original_tif.split('.')[0]
    packbit_base = "_packbit_compressed.tif"
    jpeg_base = "_jpeg_compressed.tif"
    deflate_base = "_deflate_compressed.tif"
    lzw_base = "_lzw_compressed.tif"

    command_packbits = "gdal_translate -of GTiff-co COMPRESS=PACKBITS -co\
     TILED=YES " + original_tif + " " + new_tif_base + packbit_base
    command_jpeg = "gdal_translate -co COMPRESS=JPEG -co TILED=YES " + original_tif + " " + new_tif_base + jpeg_base
    command_deflate = "gdal_translate -of GTiff -co COMPRESS=DEFLATE -co\
     PREDICTOR=" + str(predictors) + " -co TILED=YES " + original_tif + " " + new_tif_base + deflate_base
    command_lzw = "gdal_translate -of GTiff -co COMPRESS=LZW -co PREDICTOR=" + str(predictors) + " -co TILED=YES " + original_tif + " " + new_tif_base + lzw_base

    command_rm = "rm " + original_tif

    if compression_method == "JPEG":
        os.system(command_jpeg)
        os.system(command_rm)
    elif compression_method == "Packbits":
        os.system(command_packbits)
        os.system(command_rm)
    elif compression_method == "Deflate":
        os.system(command_deflate)
        os.system(command_rm)
    elif compression_method == "LZW":
        os.system(command_lzw)
        os.system(command_rm)


def retrieve_images(events, overwrite_if_exists=False):
    for date, urls in events.items():
        if date == '2017-09-05' or date == '2017-09-10': continue
        if not os.path.isdir('%s' % date):
            os.mkdir('%s' % date)
        os.chdir('%s' % date)
        print('Downloading images from {}'.format(date))
        for i, url in enumerate(urls):
            filename = url[url.rfind("/")+1:]
            compressed_file_name = filename.split('.')[0] + '_jpeg_compressed.tif'
            if overwrite_if_exists or not os.path.isfile(compressed_file_name):
                urlretrieve(url, filename)
                print('downloaded file: {}'.format(filename))
                compress_tif(os.path.join(os.getcwd(), filename))
                print('compressed file: {}'.format(filename))
        os.chdir('..')
        print()


def main():
    args = get_args()
    links = get_img_links(os.path.join(args.event_url))
    pre_events, post_events = get_img_by_date(links)
    os.chdir('/Volumes/ExtremeSSD/cs461_final_project/data/disaster_images/manually_selected/pre_event')
    print("Processing Pre-Events")
    retrieve_images(pre_events)
    os.chdir('../post_event')
    print("Processing Post-Events")
    retrieve_images(post_events)


if __name__ == "__main__":
    main()