import os
'''
Use delete.py to clear the dataset document
'''
CUR_PATH = r'./straightening_single_chromosomes'
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        for j in os.listdir(c_path):  
            new_path = os.path.join(c_path, j)  
            if os.path.isdir(new_path):
                del_file(new_path)
            else:
                os.remove(new_path)

del_file(CUR_PATH)