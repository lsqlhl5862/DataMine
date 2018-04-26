import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
plt.style.use('ggplot')


df = pd.read_csv("segment-tain.txt", skiprows=107,header=None)
df[19].replace(["brickface","sky","foliage","cement","window","path","grass"], [1,2,3,4,5,6,7],inplace=True)

print(df)