from glob import glob
import pandas as pd
import os

folderlist = os.listdir(os.path.join('../../datasets','test/data/'))

ImgidList = []
ImgFoldersList = []
for folder in folderlist:
    path =  os.path.join('../../datasets','test/data/',folder)
    listoffile = glob(path+'/*.jpg')
    for filename in listoffile:
        Imgid = (filename.split('/'))[-1].split('\\')[-1]
        ImgidList.append(Imgid)
        ImgFoldersList.append(folder)

datadict = {"id":ImgidList,"folders":ImgFoldersList}
df = pd.DataFrame(datadict)
df.to_csv('TestImageID.csv',index=False)

