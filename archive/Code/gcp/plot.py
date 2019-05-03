import os
import pandas as pd
import matplotlib.pyplot as plt

f = open("tsn_training.log","r")
lines = f.readlines()
df = list()
for line in lines[1:]:
    df.append([float(each) for each in line.strip().split(',')])

df = pd.DataFrame(df,columns = lines[0].strip().split(','))

text = "Learning Rates:\n Epochs 0-44: 1e-3\n Epochs 45-200:2e-3"

fig,ax = plt.subplots()
plt.plot(df['epoch'], df['acc'], color = 'b', label = 'Training Accuracy')
plt.plot(df['epoch'], df['val_acc'], color = 'r', label = 'Validation Accuracy')
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.title("TSN Performance")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.3, text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
fig.savefig("tsn_perf_acc.png")
plt.close()
