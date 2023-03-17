
root_dir = r'C:\Users\azure.lavdierada\UPFlow_pytorch\video_frames'
import os
train_test = os.listdir(root_dir)

import shutil
for folder in train_test:
    images_labels = os.listdir(os.path.join(root_dir, folder))
    if len(os.listdir(os.path.join( root_dir, folder, images_labels[0]) )) > len(os.listdir( os.path.join( root_dir, folder, images_labels[1]))):
        images_folder = images_labels[0]
        labels_folder = images_labels[1]
    else:
        images_folder = images_labels[1]
        labels_folder = images_labels[0]

    images_files = os.listdir(os.path.join( root_dir, folder, images_folder) )
    
    labels_files = os.listdir(os.path.join( root_dir, folder, labels_folder) )
    labels_files = [ file.split('.')[0] for file in labels_files]
    for i in range(len( os.listdir(os.path.join( root_dir, folder, images_folder) ) )):
        if images_files[i].split('.')[0] not in labels_files:
            print(images_files[i].split('.')[0])
            #shutil.move(os.path.join( root_dir, folder, images_folder, images_files[i])  , os.path.join( root_dir, folder) )




    
    

    