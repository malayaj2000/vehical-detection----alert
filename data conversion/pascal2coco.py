import xml.etree.ElementTree as ET
from glob import glob
import os 
import json

folderlist = os.listdir(os.path.join('../../datasets','test/data/'))
data = dict()
for f in folderlist:
    path =  os.path.join('../../datasets','test/data/',f)
    listoffile = glob(path+'/*.xml')
    for filename in listoffile:
        tree = ET.parse(filename)
        root = tree.getroot()
        sizenode= root.findall('size')
        width = 0
        height = 0
        depth = 0
        name = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for s in sizenode:
            width = s.find('width').text
            height = s.find('height').text
            depth = s.find('depth').text

        for obj in root.findall('object'):    
            name.append(obj.find('name').text)
            bbox = obj.find('bndbox')
            xmin.append(bbox.find('xmin').text)
            ymin.append(bbox.find('ymin').text)
            xmax.append(bbox.find('xmax').text)
            ymax.append(bbox.find('ymax').text) 

        Imgid = (filename.split('/'))[-1].split('\\')[-1]
        data[Imgid] = []
        allobjects = []
        for i in range(len(name)):
            objectList = []
            objectList.append(name[i])
            objectList.append(int(xmin[i]))
            objectList.append(int(ymin[i]))
            objectList.append(int(xmax[i])-int(xmin[i]))
            objectList.append(int(ymax[i])-int(ymin[i]))
            allobjects.append(objectList)
        data[Imgid].append(allobjects)
        # data[Imgid].append(int(width))
        # data[Imgid].append(int(height))
        # data[Imgid].append(int(depth))




with open('annotationTest.json', 'w') as fp:
    json.dump(data, fp)

#print(folderlist)
