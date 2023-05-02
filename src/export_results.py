import pickle
import csv
import numpy as np
import os
import pdb
path = '.'
for root, dirs, files in os.walk(path):
    for dir in dirs:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file == 'evaluate_record.pickle':
                    filename = dir+'/'+file
                    with open(filename,'rb') as infile:
                        data = pickle.load(infile)
                        losses = data['all_time_loss_record']
                        # pdb.set_trace()
                    with open(dir+'/'+dir+'_30xp.csv','w') as outfile:
                        csv_writer = csv.writer(outfile, delimiter=';')
                        # csv_writer.writerows(losses[0][-1])
                        for loss in losses:
                            # print(type(loss))
                            # print(len(loss))
                            csv_writer.writerow(loss[-1])