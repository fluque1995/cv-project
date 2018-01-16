import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = "accuracy_dropout.csv"

data = pd.read_csv(filename)

filename2 = "accuracy_vlr.csv"

data2 = pd.read_csv(filename2)

sns.set()
fig = sns.pointplot(x="Step",
                    y="Value",
                    data=[data, data2])

plt.xticks(rotation=60)
plt.show()
