import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = "accuracy_base.csv"

data = pd.read_csv(filename)

sns.set()
fig = sns.pointplot(x="Step",
                    y="Value",
                    data=data)

plt.xticks(rotation=60)
plt.show()
