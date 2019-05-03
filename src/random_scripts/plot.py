import os
import pandas as pd
import matplotlib.pyplot as plt

f = open("../training_logs/tsn_training.log","r")
lines = f.readlines()
df = list()
for line in lines[1:]:
    df.append([float(each) for each in line.strip().split(',')])

df = pd.DataFrame(df,columns = lines[0].strip().split(','))

text = "Learning Rates:\n Epochs 0-44: 1e-3\n Epochs 45-200:2e-3"
#text = "Learning rate: 1e-3"

fig,ax = plt.subplots()

plt.plot(df['epoch'], df['loss'], color = 'b', label = 'Training Loss')
plt.plot(df['epoch'], df['val_loss'], color = 'r', label = 'Validation Loss')

#plt.plot(df['epoch'], df['acc'], color = 'b', label = 'Training Accuracy')
#plt.plot(df['epoch'], df['val_acc'], color = 'r', label = 'Validation Accuracy')

plt.legend()

plt.ylabel("Loss")
#plt.ylabel("Accuracy")

plt.xlabel("Epochs")

plt.title("TSN Loss Curve")
#plt.title("I3D Frame Accuracy Curve")

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

fig.savefig("../results/tsn_perf_loss.png")
#fig.savefig("i3d_frame_perf_acc.png")

plt.close()
