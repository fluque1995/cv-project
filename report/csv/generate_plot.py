import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< HEAD
filename = "accuracy_dropout.csv"
=======
filename = "loss_vlr.csv"
>>>>>>> ef9d842f504a05f076c8f94a508f558d2e35463f

data = pd.read_csv(filename)

sns.set()
fig = sns.pointplot(x="Step",
                    y="Value",
                    data=data)

plt.xticks(rotation=60)
plt.show()
