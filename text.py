import numpy as np
a = np.arange(5)
keep = np.logical_and(a>1, a<4)
print(keep)
keep = []
for i in a:
    if 1 < i < 4:
        keep.append(True)
    else:
        keep.append(False)
print(keep)