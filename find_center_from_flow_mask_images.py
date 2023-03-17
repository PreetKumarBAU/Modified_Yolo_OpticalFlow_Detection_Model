import cv2
import numpy as np
import os

def inbetween(minv, val, maxv):
    return float(min(maxv, max(minv, val))) 

root_dir = r'C:\Users\azure.lavdierada\yolov5\DA'
mask_dir = os.path.join(root_dir, 'mask')
annotation_dir = os.path.join(root_dir, 'Ann')
chagas_dir = r"C:\Users\azure.lavdierada\yolov5\DA\test_chagas"
videos = os.listdir(chagas_dir)

chagas_label = 0
for video in videos:
    frames = os.listdir( os.path.join(mask_dir, video) )
    os.makedirs(os.path.join(annotation_dir, video), exist_ok= True)

    for frame in frames:
        image_path = os.path.join(mask_dir, video, frame)

        # read mask image as grayscale
        #print(image_path)
        
        img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img , (416, 416))
        
        ## Open a File in Write Mode
        
        txt_filename = video+ frame.split('.')[0] + '.txt'
        txt_file_path = os.path.join(annotation_dir, video, txt_filename)

        file1 = open(txt_file_path,"w")#Write mode

        # get shape
        height, width = img.shape

        # get contours (presumably just one around the nonzero pixels) 
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        center_coordinates_list = []
        for cntr in contours:
            area = cv2.contourArea(cntr)
            if area > 15:
                x,y,w,h = cv2.boundingRect(cntr)
                xc = x + w//2
                yc = y + h//2
                #cv2.rectangle(image, start_point, end_point, color, thickness)
                #img = cv2.rectangle(img_1,(x,y),(x+w,y+h),(0,255,0),2)


                ## Increase the width and height
                #print((h, w))

                new_h = int(h * 3)
                new_w = int( w * 1.5)
                new_x = int( x - ((new_w - w)//2))
                new_y = int( y - ((new_h - h)//2))


                x1 , y1 , x2 , y2 = new_x , new_y , new_x+new_w , new_y+new_h
                new_xc = x1 + new_w//2
                new_yc = y1 + new_h//2
                minv = 0.001
                maxv = 1.0

                #print(( chagas_label, new_xc/width , new_yc/height, new_w/width, new_h/height   ))
                #center_coordinates = (xc, yc )
                #center_coordinates_list.append(center_coordinates)

                #print((new_h , new_w))
                #img_1 = cv2.rectangle(img_1,(new_x,new_y),(new_x+new_w,new_y+new_h),(255,0,0),3)
                ## Write in the txt file
                '''
                scaled_x = xc/width
                scaled_y = yc/height
                scaled_w = w/width
                scaled_h = h/height
                '''
                scaled_x = new_xc/width
                scaled_y = new_yc/height
                scaled_w = new_w/width
                scaled_h = new_h/height
                
                file1.write("{} {} {} {} {}\n".format( chagas_label, inbetween(minv, scaled_x , maxv)  ,  inbetween(minv,scaled_y , maxv) , inbetween(minv,scaled_w , maxv) , inbetween(minv, scaled_h , maxv)   )) 

                if inbetween(minv, new_x/width, maxv)  < 0.0 or   inbetween(minv, new_y/height, maxv) < 0.0 or inbetween(minv, new_w/width, maxv) < 0.0 or  inbetween(minv, new_h/height , maxv) < 0.0 or inbetween(minv, new_x/width, maxv)  > 1.0 or   inbetween(minv, new_y/height, maxv) > 1.0 or inbetween(minv, new_w/width, maxv) > 1.0 or  inbetween(minv, new_h/height , maxv) > 1.0:
                    print(inbetween(minv, new_x/width, maxv)  ,  inbetween(minv, new_y/height, maxv) , inbetween(minv, new_w/width, maxv) , inbetween(minv, new_h/height , maxv) )
        file1.close()
        #cv2.imshow('image' , img)
        #cv2.waitKey(100000)
       
    
                #image = cv2.circle(img, center_coordinates, radius=0, color=(0, 0, 0), thickness=5)

#cv2.imshow('image' , image)

#key = cv2.waitKey(100000)
