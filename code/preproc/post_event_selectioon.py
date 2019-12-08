import argparse
import os
import requests
from urllib.parse import urlparse
from urllib.request import urlretrieve
from collections import defaultdict
from bs4 import BeautifulSoup


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


def main():
    event_url = 'https://www.digitalglobe.com/ecosystem/open-data/hurricane-maria'
    links = get_img_links(os.path.join(event_url))
    pre_events, post_events = get_img_by_date(links)
    img_names = []
    for link in list(post_events.values())[0]:
        url = urlparse(link)
        code = url.path.split('/')[5]
        name = code.split('.')[0]
        img_names.append(name + '_jpeg_compressed.tif')
    os.chdir('/Volumes/ExtremeSSD/cs461_final_project/data/disaster_images/post_event/2017-09-24/')
    for img in img_names:
        os.system('cp ' + os.path.join(os.getcwd(), img) + ' ../../manually_selected/post_event/2017-09-24/')
    # os.chdir('/Volumes/ExtremeSSD/cs461_final_project/data/disaster_images/manually_selected/post_event')
    # retrieve_images(pre_events)
    # os.chdir('../post_event')
    # print("Processing Post-Events")
    # retrieve_images(post_events)


if __name__ == "__main__":
    main()