import os
import re

def process_line_item(x):
    index = x[0]
    line_str = x[1]
    new_first_value = [str(0)] # you can put here new value depending on line index
    return ' '.join(new_first_value + x[1].split(' ')[1:])

      
root_dir = r'C:\Users\azure.lavdierada\yolov5\dataset'
for filename in os.listdir(root_dir):
    if filename.endswith(".txt"):
        txt_file = os.path.join( root_dir, filename)
        with open(txt_file, "r") as f:    
            lines = len(f.readlines())
            print(lines)
            for i in range(lines):
                line = f.readline()
                print(line)
                
                if i ==0:
                    with open(txt_file, "w") as f:
                        splt = line.split(" ")
                        f.write('0' + " " + " ".join(splt[1:]))
                else:
                    with open(txt_file, "a") as f:
                        splt = line.split(" ")
                        f.write('0' + " "  + " ".join(splt[1:])) 
        f.close()          

        '''
        data = None
        #print(txt_file)
        with open(txt_file, 'r') as file:
            lines = len(file.readlines())
            data = file.readlines()
            print(data)
            if len(data) == 0 :
                print('vvvvvvvvvvvvv') 
                continue
            else:
                print('dddd')

                new_data = '\n'.join(list(map(process_line_item, enumerate(data.split('\n')))))
                with open(txt_file, 'w') as file:
                    file.write(new_data)
        '''








