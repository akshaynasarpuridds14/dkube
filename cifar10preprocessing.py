DATA_DIR='/opt/dkube/input'

path=DATA_DIR+'/train.7z'
path_labels=DATA_DIR+'/trainLabels.csv'

import py7zr
with py7zr.SevenZipFile(path, mode='r') as z:
    z.extractall(path="/opt/dkube/output")
    
# importing shutil module
import shutil
 
# Source path
source = path_labels
 
# Destination path
destination = "/opt/dkube/output/trainLabels.csv"

shutil.copyfile(source, destination)