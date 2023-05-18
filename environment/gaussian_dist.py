import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
[
[50,60],
[60,50]
]
'''
if __name__ == '__main__':
    
    s_1 = np.random.multivariate_normal([300,300], [[50,0],[0,30]], size=400)
    s_2 = np.random.multivariate_normal([150,200], [[350,0],[0,40]], size=400)
    df = pd.DataFrame(data = s_1, columns = ['1','2'])
    print(df.cov())
    print(df.corr())
    plt.plot(s_1[:, 0], s_1[:, 1], '.', alpha=0.5)
    plt.plot(s_2[:, 0], s_2[:, 1], '.', alpha=0.5)
    plt.axis('equal')
    plt.grid()
    plt.show()

