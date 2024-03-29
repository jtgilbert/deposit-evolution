import pandas as pd

df = pd.read_csv('Inputs/SleepingChildQ_2001_2008_15min.csv')

out_datetime = []
out_q = []

for i in df.index:
    if ':00' in df.loc[i, 'Datetime']:
        out_datetime.append(df.loc[i, 'Datetime'])
        out_q.append(((df.loc[i, 'Q']+df.loc[i+1, 'Q']+df.loc[i+2, 'Q']+df.loc[i+3, 'Q'])/4)) # halving discharge

data = {'Datetime': out_datetime, 'Q': out_q}
out_df = pd.DataFrame(data=data)

out_df.to_csv('Inputs/SleepingChildQ_2001_2008_1hr.csv')
