import math
import numpy as np

f = open('custom3.txt','r')
lines = f.readlines()
f.close()
total_length = len(lines)


np.random.seed(0)
idxs = np.random.permutation(total_length)
t_idxs = idxs[:math.floor(total_length*0.8)]
test_idxs = idxs[math.floor(total_length*0.8):]
train_length = len(t_idxs)
train_idxs = t_idxs[:math.floor(train_length*0.8)]
val_idxs = t_idxs[math.floor(train_length*0.8):]

print ("Train set:", train_idxs.shape)
print ("Validation set:", val_idxs.shape)
print ("Test set:", test_idxs.shape)

f = open('custom3Train.txt','w')
f.writelines(np.array(lines)[train_idxs])
f.close()

f = open('custom3Val.txt','w')
f.writelines(np.array(lines)[val_idxs])
f.close()

f = open('custom3Test.txt','w')
f.writelines(np.array(lines)[test_idxs])
f.close()