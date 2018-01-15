import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = "loss_norm.csv"

data = pd.read_csv(filename)

sns.set()
fig = sns.pointplot(x="Step",
                    y="Value",
                    data=data[::4])

plt.xticks(rotation=60)
plt.show()
