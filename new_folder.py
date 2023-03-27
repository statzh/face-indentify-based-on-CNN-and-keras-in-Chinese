#-*- coding: utf-8 -*-

import os
import sys
from face_data import CatchPICFromVideo as CPV
def create_new_folder():
    mother_path = 'F:/face_identify/'
    new_name = str(input("Please input new file name: "))
    whole_path = os.path.abspath(os.path.join(mother_path, new_name))
    #print(whole_path)

    if os.path.exists(whole_path):
        raise ValueError("The path already existed!")
    else:
        os.mkdir(whole_path)
        print("New folder has been created!")
        return whole_path


if __name__ == '__main__':  
    if len(sys.argv) != 3:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        whole_path = create_new_folder()
        CPV("Collecting face data", int(sys.argv[1]), int(sys.argv[2]), whole_path)
