from sys import argv
from more_itertools import grouper
import csv

with open(argv[1], 'r') as f:
    fdata = f.readlines()

timetraining = fdata[-2][len("Training complete in "):]
fdata = fdata[:-2]
out = [["Time:",timetraining[:-1]],["Epoch","Train Loss","Val Loss","Train Acc","Val Acc"]]
for i, (a, b, c, d, e) in enumerate(grouper(5,fdata)):
    cs = c.split(' ')
    ds = c.split(' ')
    out.append([str(i),cs[2],ds[2],cs[4][:-1],ds[4][:-1]])

with open(argv[1]+'.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in out:
        wr.writerow(row)