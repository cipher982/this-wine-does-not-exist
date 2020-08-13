import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from requests import get
import lxml

def scrapeHTML(url):
    print(url)
    try:
        html = get("http://www.wine.com"+url).text
    except Exception as e:
        html = e
        return url, html

    return url, html


def extractDescription(html):
    try:
        soup = BeautifulSoup(html, "lxml")
        desc = soup.find("div", {"itemprop":"description"})\
                   .find("div", {"class":"viewMoreModule_text"}).text
    except Exception as e:
        desc = e

    return desc

def scrapeAndExtract(url):
    print(url.encode("ascii", errors="ignore").decode())
    try:
        html = get("http://www.wine.com"+url).text
        soup = BeautifulSoup(html, "lxml")
        desc = soup.find("div", {"itemprop":"description"})\
                   .find("div", {"class":"viewMoreModule_text"})\
                   .encode("ascii", errors="ignore").decode()
    except Exception as e:
        desc = e
        
    url = url.encode("ascii", errors="ignore").decode()
    return url, desc
