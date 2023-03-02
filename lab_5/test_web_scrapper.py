import requests
from bs4 import BeautifulSoup

url = "http://bonnat.ucd.ie/jigsaw/index.jsp?q=complex"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

aTags = soup.select(
    "body > table tr > td:nth-child(2) > table tr:nth-child(2) > td:nth-child(1) table a")

simpleElaborations = []
for aTag in aTags:
    simpleElaborations.append(aTag.text)

print(simpleElaborations)
