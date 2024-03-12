import numpy as np
import os
import matplotlib.pyplot as plt
import random

file_path = 'E:\ResearchCode\ThreeObjects\V72\DataFile'
file_list = os.listdir(file_path)

w = [] # all the weight values
for item in file_list:
    w1 = float(item.split('_')[3])
    w2 = float(item.split('_')[5])
    w.append([w1,w2,1-w1-w2])
w = np.array(w)
print(len(w))

train_w = w


train_w = np.array(train_w)

print('the length of train_w: ',len(train_w))


plt.scatter(train_w[:,0],train_w[:,1])
plt.show()

np.save('train_ws_w.npy',train_w)




