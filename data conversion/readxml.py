import xml.etree.ElementTree as ET

tree = ET.parse('../../datasets/train/data/train_v1/frame_579_jpg.rf.5e046cbdc8db6475903b901b3a0372c3.xml')

root = tree.getroot()

sizenode= root.findall('size')
width = 0
height = 0
depth = 0
for s in sizenode:
    width = s.find('width').text
    height = s.find('height').text
    depth = s.find('depth').text

name = []
xmin = []
ymin = []
xmax = []
ymax = []

for obj in root.findall('object'):    
    name.append(obj.find('name').text)
    bbox = obj.find('bndbox')
    xmin = bbox.find('xmin').text
    ymin = bbox.find('ymin').text
    xmax = bbox.find('xmax').text
    ymax = bbox.find('ymax').text

print(name)

