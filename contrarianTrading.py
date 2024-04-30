import numpy as np
import pandas as pd

class contrarianTrader():

    def __init__(self):
        pass

    @staticmethod
    def top_bottom_50(self, umd, data, filter_1, filter_2, K=1):
        
        umd = umd.groupby('date').apply(self.assign_momr)
        umd.reset_index(inplace=True, drop=True)
        umd['momr']=umd.momr.astype(int)
        umd['momr'] = umd['momr']+1 #indexed at 0 so increase by 1 
        umd['form_date'] = umd['date']
        umd['medate'] = umd['date']+MonthEnd(0)
        umd['hdate1']= umd['medate']+MonthBegin(1)
        umd['hdate2']= umd['medate']+MonthEnd(K)
        umd = umd[['compNam', 'form_date','momr','hdate1','hdate2']]
        
        _tmp_ret = data[['compNam','date','logret']]
        _tmp_ret['ret'] = np.exp(_tmp_ret.logret)-1 #unlogging the return 
    
        #joining the ranking and return dataframes
        #chunking the join to reduce RAM load
        
        chunk_size = 10000  
        chunks = [umd[i:i+chunk_size] for i in range(0, umd.shape[0], chunk_size)]

        port = pd.DataFrame()
        
        for chunk in tqdm(chunks):
            
            merged_chunk = pd.merge(_tmp_ret, chunk, on=['compNam'], how='inner', sort=False)
            port = pd.concat([port, merged_chunk], ignore_index=True)
            port = port[(port['hdate1']<=port['date']) & (port['date']<=port['hdate2'])]
            
        
        umd2 = port.sort_values(by=['date','momr','form_date','compNam']).drop_duplicates()
        umd3 = umd2.groupby(['date','momr','form_date'])['ret'].mean().reset_index()

        # reduce sample based on size 
        umd3 = umd3[umd3['date'] < pd.to_datetime(filter_2)]
        umd3 = umd3[umd3['date'] >= pd.to_datetime(filter_1)]
        umd3 = umd3.sort_values(by=['date','momr'])
        
        # Create one return series per MOM group every month
        ewret = umd3.groupby(['date','momr'])['ret'].mean().reset_index()
        ewstd = umd3.groupby(['date','momr'])['ret'].std().reset_index()
        ewret = ewret.rename(columns={'ret':'ewret'})
        ewstd = ewstd.rename(columns={'ret':'ewretstd'})
        ewretdat = pd.merge(ewret, ewstd, on=['date','momr'], how='inner')
        ewretdat = ewretdat.sort_values(by=['momr'])
        
        #summarising the portfolio returns
        decileRets = ewretdat.groupby(['momr'])['ewret'].describe()[['count','mean', 'std']]
        
        # Transpose portfolio layout to have columns as portfolio returns
        ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ewret')
    
        # Add prefix port in front of each column
        ewretdat2 = ewretdat2.add_prefix('port')
        ewretdat2 = ewretdat2.rename(columns={'port3' :'winners', 'port1':'losers', 'port2' : 'middle'})
        ewretdat2['long_short'] = ewretdat2['losers'] - ewretdat2['winners']

        # Compute Long-Short Portfolio Cumulative Returns
        ewretdat3 = ewretdat2
        ewretdat3['1+losers']=1+ewretdat3['losers']
        ewretdat3['1+winners']=1+ewretdat3['winners']
        ewretdat3['1+ls'] = 1+ewretdat3['long_short']

        ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
        ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
        ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1

        #################################
        # Portfolio Summary Statistics  #
        ################################# 

        # Mean 
        mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
        mom_mean = mom_mean.rename(columns={0:'mean'}).reset_index()

        # T-Value and P-Value
        t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
        t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
        t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T

        t_losers['momr']='losers'
        t_winners['momr']='winners'
        t_long_short['momr']='long_short'

        t_output =pd.concat([t_winners, t_losers, t_long_short])\
            .rename(columns={0:'t-stat', 1:'p-value'})

        # Combine mean, t and p
        mom_output1 = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
        
        return decileRets, mom_output1, ewretdat2, port, ewretdat, ewretdat3

    @staticmethod
    def assign_momr(self, group):
        
        mean_ret = group['rawRet'].mean()
        
        group['rawRet'] = [i if i > -1 else 0 for i in group['rawRet'].tolist()] #this line stops the stocks that were delisted in the previous period from being included in the 'bottom 50' portfolio 
        
        # Sort the group by 'ret' to identify top and bottom values
        sorted_group = group.sort_values(by='rawRet')

        # Get the top and bottom 50 values
        top_50_values = sorted_group['rawRet'].tail(50)
        bottom_50_values = sorted_group['rawRet'].head(50)

        # Function to assign momr value
        def get_momr_value(row):
            if row['rawRet'] in bottom_50_values.values:
                return 0
            elif row['rawRet'] in top_50_values.values:
                return 2
            else:
                return 1

        # Apply the function to assign 'momr'
        group['momr'] = group.apply(get_momr_value, axis=1)

        return group