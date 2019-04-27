import pandas as pd
import matplotlib.pyplot as plt

f = open("tsn_training.log","r")
lines = f.readlines()

stack = list()
columns = lines[0].strip().split(',')

for line in lines[1:]:
    line = line.strip()
    stack.append([float(x) for x in line.split(',')])

df = pd.DataFrame(stack,columns = columns)

fig, ax = plt.subplots()

plt.plot(df['epoch'].values, df['acc'].values, color ='b', label = 'Training Loss')
plt.plot(df['epoch'].values, df['val_acc'].values, color ='r', label = 'Validation Loss')
plt.title('TSN Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
fig.savefig('tsn_acc.png')
plt.close()