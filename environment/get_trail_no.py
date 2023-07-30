#the file is used to get the total no of trails for all the participants

import pandas as pd
import numpy as np
import os
import pdb
import scipy


def func_gr(gravity):
    
    #gravity = 200

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
        all_df = pd.DataFrame(columns = [
            'gravity', 'game_name', 'pvalue', 'tvalue',
            'avg_attempts_model', 'avg_attempts_participant'
        ])
        
    for fil in files:
        df = pd.read_csv(
        path_alf+fil
        )
        df_sub_part = df_part[
            (df_part['Gravity']==gravity) & (df_part['game_name']==fil.split('.csv')[0])
        ]

        part_attempt = df_sub_part.groupby("participant ID").agg({"attempt #":max}).reset_index()
        part_attempt.loc[
            part_attempt[part_attempt['attempt #']>=11].index, "attempt #"
        ] = 10
        try:
            model_attempt = df.groupby("game iter").agg({'total_attempt':max}).reset_index()
        except:
            pdb.set_trace()
        model_attempt.loc[model_attempt[
            model_attempt['total_attempt']==11].index, 'total_attempt'] = 10
        t_stat = scipy.stats.ttest_ind(
            part_attempt["attempt #"],
            model_attempt['total_attempt'].sample(len(part_attempt)))

        all_df = pd.concat([all_df, pd.DataFrame([
            {'gravity':gravity, 'game_name':fil.split('.csv')[0], 'pvalue':round(t_stat.pvalue,4),
             'tvalue': round(t_stat.statistic,4), 'avg_attempts_model':model_attempt['total_attempt'].mean(), 'avg_attempts_participant': part_attempt["attempt #"].mean()}
        ])], ignore_index=True)
        
        print("The t stat for the game ", fil, " is -->>", round(t_stat.pvalue,4), "\n")
        #df.groupby().reset_index()
    pdb.set_trace()
    all_df.to_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/pvalue.csv',
        index=False
    )
    
        
        #all_df = pd.concat([all_df, df])



if __name__ == '__main__':
    func_gr(200)
    func_gr(100)
    func_gr(400)
