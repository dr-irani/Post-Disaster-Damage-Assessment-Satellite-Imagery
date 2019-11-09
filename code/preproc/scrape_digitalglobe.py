import argparse
import os
import requests
from urllib.parse import urlparse
from collections import defaultdict
from bs4 import BeautifulSoup


def get_args():
    parser = argparse.ArgumentParser(description="Scrape GeoTIFF images from DigitalGlobe.")
    parser.add_argument('event_url', metavar='link to event on DigitalGlobe', type=str)
    parser.add_argument('--download', metavar='testing file', default=False, type=bool)
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

        if 'pre-event' in event: pre_events[date].append(link)
        elif 'post-event' in event: post_events[date].append(link)

    return pre_events, post_events


def main():
    args = get_args()
    links = get_img_links(os.path.join(args.event_url))
    pre_events, post_events = get_img_by_date(links)
    
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()