import os

PATH = './straightening_single_chromosomes'
f = open('./all_label.txt','w')

for k in os.listdir(PATH + '/'):
    i = PATH + '/' + k
    for j in os.listdir(i + '/'):
        label = j.split('_')[1].split('.')[0][0:len(j.split('_')[1].split('.')[0])-1]
        output = j + '\t' + label + '\n'
        f.write(output)
