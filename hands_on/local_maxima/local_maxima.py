import numpy as np

def find_maxima(a):
    print(np.diff(a).shape)
    b = np.concatenate((np.diff(a), np.array([-1])))

    max_id = []
    old = 1
    for n, el in enumerate(b):
        if old>=0 and el<0:
            max_id.append(n)
        old = el

    return max_id

if __name__ == '__main__':
    a = [1,2,2,1]
    print(find_maxima(a))
