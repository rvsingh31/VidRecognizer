import os

f = open('custom3.txt','r')
lines = f.readlines()
f.close()

parent = "FramesFlows/custom3"

out = open("error.txt","w")

for each in lines[:10]:
    path1 = os.path.join(parent,each.split(" ")[0])
    path2 = os.path.join(path1, "info.txt")
    if os.path.exists(path1) and os.path.exists(path2):
        continue
    else:
        out.write(each)
        out.write("\n")

out.close()