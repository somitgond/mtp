import numpy as np
import seaborn as sns
data = np.load("pairwise_sync_rate_1000_to_2000.npy")

for i in range(len(data)):
    pt = sns.heatmap(data[i])
    f = pt.get_figure()
    f.savefig(f"images/{i}.png")
    f.clf()
    print(f"images/{i}.png")

