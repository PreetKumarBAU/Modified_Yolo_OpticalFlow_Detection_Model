import numpy as np
import os
import pathlib
import shutil
import os
import csv
#root_dir = r"C:\Users\azure.lavdierada\Downloads\dense-ulearn-vos-main\dense-ulearn-vos-main\data\chagas_train"

#root_dir = r"C:\Users\azure.lavdierada\Structural_Similiarity_UNet_KNN\ChagasVideos\val_images"
root_dir = r"C:\Users\azure.lavdierada\ARFlow1\DATA\flow_dataset\MPI_Sintel\training"
base_name = os.path.basename(root_dir)


## Open a File in Append Mode
csv_filename = "test.csv"
file1 = open(csv_filename,"w", newline='')#append mode
writer = csv.writer(file1)
flag = 1

list_folders = os.listdir(root_dir)
#images_folder_name = list_folders[1]
#annotations_folder_name = list_folders[0]

images_folder_name = "test_chagas"
annotations_folder_name = "mask"
annotations_folder_name1 = "Annotations"
#print( os.listdir( os.path.join( root_dir , annotations_folder_name ) ))

Number_Image_Folders = len( os.listdir( os.path.join( root_dir , images_folder_name ) ))
Number_Annotation_Folders = len( os.listdir( os.path.join( root_dir , annotations_folder_name ) ) )
#Number_Annotation_Folders = 0

if Number_Annotation_Folders == 0:
    list_videos_annofolder = ['None'] * Number_Image_Folders
else:
    list_videos_annofolder = sorted( os.listdir( os.path.join( root_dir , annotations_folder_name ) ) )

#print(list_videos_annofolder)

list_videos_imagefolder = sorted(os.listdir( os.path.join( root_dir , images_folder_name ) ) )

print("list_videos_imagefolder:::",list_videos_imagefolder)

for (video_name_Jpeg , video_name_anno) in zip(list_videos_imagefolder, list_videos_annofolder ):
    
    image_filenames = sorted ( os.listdir( os.path.join( root_dir , images_folder_name , video_name_Jpeg )) )
    
    num_image_frames = len(image_filenames)
    if Number_Annotation_Folders == 0:
        anno_filenames = ['None'] * num_image_frames
    else:
        anno_filenames = sorted ( os.listdir( os.path.join( root_dir , annotations_folder_name , video_name_Jpeg )) )

    print('video name',video_name_Jpeg)
    print( "Number of image_filenames:::", len( image_filenames) )
    print( "Number of anno_filenames:::", len(anno_filenames) )
    frame_number = []
    
    for i, (image_filename, anno_filename) in enumerate( zip(image_filenames[:-1] , anno_filenames[:-1] )):

        frame_number.append( int( image_filename.split(".")[0].split("frame_")[1] ) )
        
        anno_filename = anno_filename.split(".")[0] + '.txt'
        image_path = os.path.join( images_folder_name, video_name_Jpeg ,  image_filename)
        anno_path = os.path.join( annotations_folder_name1, video_name_Jpeg ,  anno_filename)
        image_path = image_path.replace(  '\\' , '/' )
        anno_path = anno_path.replace(  '\\' , '/' )
        #print(image_path)
        #print(anno_path)
        
        #print("image_path::" ,image_path )
        #image_path_ = "./data/" + image_path
    
    #start_frame_number = min(frame_number)
    #end_frame_number   = max(frame_number)
        row = [image_path , anno_path]
        writer.writerow(row)
        #file1.write("{} {}\n".format( image_path , anno_path  ))     
    '''
        if Number_Annotation_Folders == 0:
            file1.write("{} \n".format( image_path  ))
            

        else:
            
            anno_path = os.path.join( base_name, annotations_folder_name, video_name_anno ,  anno_filename )
            anno_path = anno_path.replace(  '\\' , '/' ) 
            #print(anno_path)
            if 'png' in anno_path:
                
                anno_path = anno_path.replace(  'png' , 'jpg' ) 
                #print("True")
            
            #anno_path_ = "./data/" + anno_path

        #print(image_path_)
        #print("EXECUTED")
            file1.write("{} {} {}\n".format(flag, image_path_ ,anno_path_ ))
            #print(image_path_)
            #file1.write("{} {}\n".format( image_path ,anno_path ))
    ''' 

        
        
        #if i == 9:
        #    break
    

        #print("path:" , path)
        
    

file1.close()


### Read File after Appending

file1 = open(csv_filename,"r")

if csv_filename in os.listdir(root_dir):
    os.remove( os.path.join(root_dir ,csv_filename ))
    shutil.copy( os.path.join(  r"C:\Users\azure.lavdierada\YoloV3" , csv_filename) , root_dir)
else:
    shutil.copy( os.path.join(  r"C:\Users\azure.lavdierada\YoloV3" , csv_filename) , root_dir)

print("Output of Readlines after appending") 
#print(file1.readlines())
file1.close()




