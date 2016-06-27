from math import sqrt

def quadratic_map(x):
    # feature map for polynomial kernel (gamma* u`v + c)^2
    # assume gamma=1, c = 0
    n = len(x)
    r = []
    r.extend([x[i] * x[i] for i in range(n - 1, -1, -1)])
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            r.append(sqrt(2) * x[i] * x[j])
    return r