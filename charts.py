import os
import pandas as pd
import matplotlib.pyplot as plt

filepath = os.path.join(os.path.dirname(__file__), 'Outputs/Woods_out_15min.csv')
df = pd.read_csv(filepath, index_col=[0, 1])

elev_upstream = [df.loc[('deposit_upstream', x), 'elev'] for x in df.index.levels[1]]
elev_downstream = [df.loc[('deposit_downstream', x), 'elev'] for x in df.index.levels[1]]

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(df.index.levels[1], elev_upstream)
ax.plot(df.index.levels[1], elev_downstream)

plt.show()

