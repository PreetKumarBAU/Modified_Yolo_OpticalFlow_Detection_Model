
import numpy as np
import os

root_dir = r'C:\Users\azure.lavdierada\ARFlow1\DATA\flow_dataset\MPI_Sintel\training\Annotations'

for video in os.listdir(root_dir):
    for txtfile in os.listdir( os.path.join(root_dir,video )):
        values = np.loadtxt(fname=os.path.join(root_dir,video , txtfile) , delimiter=" ", ndmin=2).tolist()
        #print(values)
        for i in range(len(values)):
            for val in values[i]:
                if (val < 0.0 or val > 1.0):
                    print("WRONG VALUES")
                    print(os.path.join(root_dir,video , txtfile))
