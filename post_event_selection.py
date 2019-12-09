import argparse
import os
import requests
from urllib.parse import urlparse
from urllib.request import urlretrieve
from collections import defaultdict
from bs4 import BeautifulSoup
from collections import defaultdict



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
        codes = ['103001005F73E000', '1030010060C0B000', '103001006FBA7400', '1030010072823700']
        if 'pre-event' in event and code in codes: pre_events[date].append(link)
        elif 'post-event' in event and code == '1030010072069C00': post_events[date].append(link)

    return pre_events, post_events


def main():
    event_url = 'https://www.digitalglobe.com/ecosystem/open-data/hurricane-maria'
    links = get_img_links(os.path.join(event_url))
    pre_events, post_events = get_img_by_date(links)
    images = defaultdict(list)
    for date, img_names in pre_events.items():
        print(len(img_names))
        for link in img_names:
            url = urlparse(link)
            code = url.path.split('/')[5]
            name = code.split('.')[0]
            images[date].append(name + '_jpeg_compressed.tif')
    count = 0
    for date, img_names in images.items():

        os.chdir(os.path.join('/Volumes/ExtremeSSD/cs461_final_project/data/disaster_images/pre_event/', date))
        for img in img_names:
            count += 1
            print(count, img)
            os.system('cp ' + os.path.join(os.getcwd(), img) + ' ../../manually_selected/pre_event/charlotte/')


if __name__ == "__main__":
    main()