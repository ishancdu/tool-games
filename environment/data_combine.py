# the script combines all the data and calculates the statistics for the values all games
import pandas as pd
import pdb
import os


def calc_stat(df):
    acc = 100 * (1-len(df[df['success']==False]['game iter'].unique())/50)
    df['total_attempt'].iloc[df[df['total_attempt'] == 11].index] = 10
    tries = df['total_attempt'].mean()

    ddf = pd.DataFrame({'accuracy':acc, 'average_tries':tries}, index=[0])
    return ddf

if __name__ == '__main__':
    
    #df = pd.read_csv(
    #    '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/output_normal/'
    #)
    path_alf = '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/output_double/'
    files = os.listdir(
        path_alf
    )
    pdb.set_trace()
    all_df = pd.DataFrame()
    for fil in files:
        df = pd.read_csv(
        path_alf+fil
        )
        all_df = pd.concat([all_df, df])
        
    all_df.to_csv(path_alf+"combined_res.csv", index=False)
    stats_df = all_df.groupby('game_name').apply(calc_stat).reset_index()
    stats_df = stats_df.drop(columns = 'level_1')
    stats_df.to_csv(path_alf+"combined_res_stats.csv", index=False)
    pdb.set_trace()
    



