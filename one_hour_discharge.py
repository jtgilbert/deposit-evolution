import pandas as pd

df = pd.read_csv('Inputs/WoodsQ_2012_2019_15min.csv')

out_datetime = []
out_q = []

for i in df.index:
    if ':00' in df.loc[i, 'Datetime']:
        out_datetime.append(df.loc[i, 'Datetime'])
        out_q.append(((df.loc[i, 'Q']+df.loc[i+1, 'Q']+df.loc[i+2, 'Q']+df.loc[i+3, 'Q'])/4)) # halving discharge

data = {'Datetime': out_datetime, 'Q': out_q}
out_df = pd.DataFrame(data=data)

out_df.to_csv('Inputs/Woods_Q_2012_2019_1hr.csv')
