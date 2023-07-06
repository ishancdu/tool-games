#the file is used to get the total no of trails for all the participants

import pandas as pd
import numpy as np
import os
import pdb
import scipy


if __name__ == '__main__':
    
    gravity = 200

    if gravity == 400:
        path_alf = '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/output_double/'
    elif gravity == 100:
        path_alf = '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/output_half/'

    elif gravity == 200:
        path_alf = '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/output_normal/'
    path_part = '/home/ishan/workspace/msc_dissertation/tool-games/environment/data/HumanDataForComparison.csv'

    files = os.listdir(
        path_alf
    )
    df_part = pd.read_csv(path_part)
    try:
        all_df = pd.read_csv(
            '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/pvalue.csv'
        )
    except FileNotFoundError:
        all_df = pd.DataFrame(columns = ['gravity', 'game_name', 'pvalue'])
        
    for fil in files:
        df = pd.read_csv(
        path_alf+fil
        )
        df_sub_part = df_part[
            (df_part['Gravity']==gravity) & (df_part['game_name']==fil.split('.csv')[0])
        ]

        part_attempt = df_sub_part.groupby("participant ID").agg({"attempt #":max}).reset_index()
        part_attempt.loc[
            part_attempt[part_attempt['attempt #']>11].index, "attempt #"
        ] = 11
        try:
            model_attempt = df.groupby("game iter").agg({'total_attempt':max})
        except:
            pdb.set_trace()
        t_stat = scipy.stats.ttest_ind(
            part_attempt["attempt #"],
            model_attempt['total_attempt'].sample(len(part_attempt)))
        
        all_df = pd.concat([all_df, pd.DataFrame([
            {'gravity':gravity, 'game_name':fil.split('.csv')[0], 'pvalue':round(t_stat.pvalue,4)}
        ])], ignore_index=True)
        
        print("The t stat for the game ", fil, " is -->>", round(t_stat.pvalue,4), "\n")
        #df.groupby().reset_index()

    all_df.to_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/pvalue.csv',
        index=False
    )
    
        
        #all_df = pd.concat([all_df, df])



