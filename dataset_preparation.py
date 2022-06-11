
import os
import cv2

# Mergng all folders from FFHQ dataset
path = "" 
for i in os.scandir(path):
     if i.is_dir():
        for each_file in Path(i.path).glob('**/*'): # grabs all files
            trg_path = each_file.parent.parent # gets the parent of the folder 
            each_file.rename(trg_path.joinpath(each_file.name)) # moves to parent folder.

# Resizing original images
train_dir = "original"
for img in os.listdir("original/"):
    img_array = cv2.imread("original/" + img)
    
    img_array = cv2.resize(img_array, (256,256))
    lr_img_array = cv2.resize(img_array, (64,64))
    cv2.imwrite("hr/" + img, img_array)
    cv2.imwrite("lr/" + img, lr_img_array)
    

# Renaming images seqentially
folderPath = r'original'

fileSequence = 0

for filename in os.listdir(folderPath):
    os.rename(folderPath + '//' + filename, folderPath + '//' + str(fileSequence) + '.png')
    fileSequence +=1
    