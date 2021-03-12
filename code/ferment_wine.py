import pandas as pd
import bs4
import glob
import re
import sys
import tqdm
import traceback
import sys

def ferment_wine(file_path):
  with open(file_path, 'r', encoding='utf8') as f:
    try:
      soup = bs4.BeautifulSoup(f, parser='lxml')

      # Extract name
      wine_name = soup.find('h1',attrs={'itemprop':'name'}).text
      skip_names = ['gift','set','pack','tasting']
      if any(x in wine_name.lower() for x in skip_names):
        return

      # Extract description
      wine_description = soup.find('div', attrs={'class':'viewMoreModule_text'})
      if len(wine_description.find_all('p')) > 0:
        wine_description = '\n'.join(str(tag.text) for tag in wine_description.find_all('p'))
      else:
        wine_description = wine_description.text

      # Extract price
      wine_price = soup.find('div', attrs={'class','productPrice'})
      if wine_price is None:
        wine_price = soup.find('meta', attrs={'itemprop':'price'})['content']
      else:
        wine_price = wine_price.find('meta', {'itemprop':'price'})['content']

      # Extract 2 levels of categories
      wine_category_1 = soup.find("ul", attrs={"class":"prodAttr"}).find("li")['title']
      wine_category_2 = soup.find("span", {"class":"prodItemInfo_varietal"}).text

      # Extract origin
      wine_origin = soup.find("span", {"class":"prodItemInfo_originText"}).text

      # Extract size
      wine_size_value = soup.find("span", attrs={"class":"prodAlcoholVolume_text"}).text
      wine_size_units = re.findall("[a-zA-Z]+", soup.find("span", attrs={"class":"prodAlcoholVolume"}).text)[0]

      # Extract ABV
      wine_abv = soup.find("span", attrs={"class":"prodAlcoholPercent_percent"}).text
      
      wine_row = pd.DataFrame({
        'name':wine_name,
        'description':wine_description,
        'price':wine_price,
        'category_1':wine_category_1,
        'category_2':wine_category_2,
        'origin':wine_origin,
        'wine_size_value':wine_size_value,
        'wine_size_units':wine_size_units,
        'wine_abv':wine_abv
        },
        index=[0]
      )
  
      return wine_row
      
    except:
      e = sys.exc_info()[0]
      print(file_path, traceback.format_exc())
      