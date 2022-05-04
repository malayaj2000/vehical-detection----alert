import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
import os 
import json


folderlist = os.listdir(os.path.join('../../datasets','train/data/'))
data = dict()
name = set()
for f in folderlist:
    path =  os.path.join('../../datasets','train/data/',f)
    listoffile = glob(path+'/*.xml')
    for filename in listoffile:
        tree = ET.parse(filename)
        root = tree.getroot()
        for obj in root.findall('object'):    
            name.add(obj.find('name').text)


data["classes"] = list(name)
df = pd.DataFrame(data)
df.to_csv('classes.csv',index=False)

#print(folderlist)
