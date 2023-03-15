# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 02:07:40 2023

@author: Pooja
"""
#Data
#Data Types
name=input("Enter your name")
num=input("Enter your age")

print(type(num),type(name))
num1=int(num)
num2=float(num1)
name1=int(name)

#string
#integer
#float



from numpy import loadtxt
path = r"C:\Users\Pooja\iris.txt"
datapath= open(path, 'r')
data_num = loadtxt(datapath, delimiter=",")
datapath.close()

import numpy as np
import csv
path = r"C:\Users\Pooja\student-mat.csv"
with open(path,'r') as f:
   reader = csv.reader(f,delimiter = ',')
   headers = next(reader)
   data_csv = list(reader)
   data_csv_num=np.asarray(data_csv)
   
import pandas as pd
path = r"C:\Users\Pooja\Iris1.xls"
data = pd.read_excel(path)


import pandas as pd
path = r"C:\Users\Pooja\student-mat.csv"
data_pd = pd.read_csv(path)

import pandas as pd
url="https://www.w3schools.com/python/pandas/data.js"
data_jsn1=pd.read_json(url)



import requests
import json
covid_data = requests.get('https://api.covid19india.org/state_district_wise.json')
#print(covid_data.status_code)
data = covid_data.text
parse_json = json.loads(data)
active_case = parse_json['Andhra Pradesh']['districtData']['Guntur']['active']
print("Active cases in Andhra Pradesh, Guntur district:", active_case)

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:15:38 2023

@author: Pooja
"""
import requests
from bs4 import BeautifulSoup

page = requests.get("https://www.dlithe.com")
page.status_code


soup = BeautifulSoup(page.content, 'html.parser')

title = soup.title.text 

# gets you the text of the <title>(...)</title>
# Extract body of page
page_body = soup.body

# Extract head of page
page_head = soup.head

# print the result
print(page_body, page_head)

# Extract first <h1>(...)</h1> text
first_h1 = soup.select('h1')[0].text
# Create all_h1_tags as empty list
all_h1_tags = []

# Set all_h1_tags to all h1 tags of the soup
for element in soup.select('h1'):
    all_h1_tags.append(element.text)
    
    
seventh_p_text = soup.select('p')[6].text

# Create top_items as empty list
image_data = []

# Extract and store in top_items according to instructions on the left
images = soup.select('img')
for image in images:
    src = image.get('src')
    alt = image.get('alt')
    image_data.append({"src": src, "alt": alt})

print(image_data)


# Create top_items as empty list
all_links = []

# Extract and store in top_items according to instructions on the left
links = soup.select('a')
for ahref in links:
    text = ahref.text
    text = text.strip() if text is not None else ''

    href = ahref.get('href')
    href = href.strip() if href is not None else ''
    all_links.append({"href": href, "text": text})

print(all_links)

#https://figshare.com/articles/dataset/smile_annotations_final_csv/3187909?file=4988956
#https://medium.com/@umerfarooq_26378/python-for-pdf-ef0fac2808b0
#https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
#https://www.freecodecamp.org/news/web-scraping-python-tutorial-how-to-scrape-data-from-a-website/
#string
"""text = 'We have kept our pace with Industry. 
    We love to work with the young\xa0 
    generation. We believe Intelligent 
    Quotient, Emotional Quotient, Domain 
    & Technology quotient are equally 
    important, we transform resource 
    competence.'"""
s1=text
print(s1)
#length of string
print(len(s1))
"""
1. sequence 
2. condition if else, if, 
3. repeat iteration loop for while
"""

for i in s1:
    print(i, end= " | ")
#to fetch words from the sentence
s1_list=s1.split()

for i in s1_list:
    print(i)

s1[-1]='a' #strings are immutable
#indexing and accessing elements of a string
s1[9]
s1[19]
s1[20]
#slicing
sub_s1=s1[9:20]