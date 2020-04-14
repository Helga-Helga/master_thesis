import sys

filename = sys.argv[1]
data = []
with open(filename) as f:
    for line in f:
        fields = line.split()
        rowdata = map(float, fields)
        data.extend(rowdata)

print(sum(data) / len(data))
