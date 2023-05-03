import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

db = np.loadtxt('./diabete.csv', dtype=np.float32, encoding='utf-8')
plt.hist(db[:, -2], bins=61, color='mediumturquoise', edgecolor='gray')
plt.xlabel('Age(year)')
plt.ylabel('Number of samples')
plt.show()
