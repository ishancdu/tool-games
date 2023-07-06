import pandas as pd
import numpy as np
import pdb

'''
def calc_stat(df):
    acc = 100 * (1-len(df[df['success']==False]['game iter'].unique())/50)
    df['total_attempt'].iloc[df[df['total_attempt'] == 11].index] = 10
    tries = df['total_attempt'].mean()

    ddf = pd.DataFrame({'accuracy':acc, 'average_tries':tries}, index=[0])
    return ddf
'''

def calc_stat(df):
    accuracy = (len(df[(df['success']==1) & (df['attempt #']<=10)][
        "participant ID"
    ].unique())/len(df[
        "participant ID"].unique()))*100
    
    df_d = df.drop(df[(df["attempt #"]>10)].index)
    tries = df_d["attempt #"].mean()
    
    ddf = pd.DataFrame({'accuracy':accuracy, 'average_tries':tries}, index=[0])
    return ddf

def calc_cumulative(df):
    """
    Gets teh cumulative solution rate over 10 tries
    """
    total_part = len(df['participant ID'].unique())
    df_cumu = pd.DataFrame(columns = ['tries', 'accuracy'])
    for trn in range(1,11):
        accuracy = (len(df[(df['success']==1) & (df['attempt #']<=trn)][
            'participant ID'].unique())/total_part)*100
        df_cumu.loc[len(df_cumu)] = {'tries':trn, 'accuracy':accuracy}
        #df_cumu = pd.concat(
        #    [df_cumu, pd.DataFrame({'tries':trn, 'accuracy':accuracy}, index=)],
        #    ignore_index=True)

    return df_cumu

def calc_cumu_mod(df):
    """
    Gets teh cumulative solution rate over 10 tries
    """

    total_part = len(df['game iter'].unique())
    df_cumu = pd.DataFrame(columns = ['tries', 'accuracy'])
    for trn in range(1,11):
        accuracy = (len(df[(df['success']==True) & (df['total_attempt']<=trn)][
            'game iter'].unique())/total_part)*100
        df_cumu.loc[len(df_cumu)] = {'tries':trn, 'accuracy':accuracy}
        #df_cumu = pd.concat(
        #    [df_cumu, pd.DataFrame({'tries':trn, 'accuracy':accuracy}, index=)],
        #    ignore_index=True)

    return df_cumu


if __name__ == '__main__':
    df = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/data/HumanDataForComparison.csv'
    )
    
    normal = [
        'Falling_v2', 'Balance', 'Falling_B', 'Launch_B', 'Shove', 'Table_B', 'Table_v2', 'Unsupport'
    ]
    df_norm = df[(df['Gravity']==200) & df['game_name'].isin(normal)]
    
    stats_norm = df_norm.groupby('game_name').apply(calc_stat).reset_index()
    stats_norm = stats_norm.drop(columns = 'level_1')
    
    stats_norm.to_csv(
        "/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants"+\
        "/combined_res_norm_part.csv", index=False)


    stats_cumu = df_norm.groupby('game_name').apply(calc_cumulative).reset_index()
    stats_cumu = stats_cumu.drop(columns = 'level_1')
    stats_cumu.to_csv(
        "/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants"+\
        "/cumulative_res_norm_part.csv", index=False)


    df = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/data/HumanDataForComparison.csv'
    )
    
    half = [
        'Bridge', 'Catapult', 'Chaining', 'Collapse', 'Launch_A', 'Remove', 'SeeSaw', 'Table_A'
    ]
    
    df_half = df[(df['Gravity']==100) & df['game_name'].isin(half)]
    
    stats_half = df_half.groupby('game_name').apply(calc_stat).reset_index()
    stats_half = stats_half.drop(columns = 'level_1')
    
    stats_half.to_csv(
        "/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants"+\
        "/combined_res_half_part.csv", index=False)

    stats_half_cumu = df_half.groupby('game_name').apply(calc_cumulative).reset_index()
    stats_half_cumu = stats_half_cumu.drop(columns = 'level_1')
    stats_half_cumu.to_csv(
        "/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants"+\
        "/cumulative_res_half_part.csv", index=False)
    
    
    
    df = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/data/HumanDataForComparison.csv'
    )
    
    doub = [
        'Bridge', 'Catapult', 'Chaining', 'Collapse', 'Launch_A', 'Remove', 'SeeSaw', 'Table_A'
    ]
    
    df_doub = df[(df['Gravity']==400) & df['game_name'].isin(doub)]
    
    stats_doub = df_doub.groupby('game_name').apply(calc_stat).reset_index()
    stats_doub = stats_doub.drop(columns = 'level_1')
    
    stats_doub.to_csv(
        "/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants"+\
        "/combined_res_double_part.csv", index=False)

    
    stats_doub_cumu = df_doub.groupby('game_name').apply(calc_cumulative).reset_index()
    stats_doub_cumu = stats_doub_cumu.drop(columns = 'level_1')
    stats_doub_cumu.to_csv(
        "/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants"+\
        "/cumulative_res_doub_part.csv", index=False)


    #cumulative for the model data
    #read model data

    mod_doub = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/result_double_gravity.csv')
    smod_doub_cumu = mod_doub.groupby('game_name').apply(calc_cumu_mod).reset_index()
    smod_doub_cumu = smod_doub_cumu.drop(columns = 'level_1')
    smod_doub_cumu.to_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_doub_cumu.csv')
    
    mod_norm = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/results_normal_gravity.csv')
    smod_norm_cumu = mod_norm.groupby('game_name').apply(calc_cumu_mod).reset_index()
    smod_norm_cumu = smod_norm_cumu.drop(columns = 'level_1')
    smod_norm_cumu.to_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_norm_cumu.csv')
    
    mod_half = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/result_half_gravity.csv')
    smod_half_cumu = mod_half.groupby('game_name').apply(calc_cumu_mod).reset_index()
    smod_half_cumu = smod_half_cumu.drop(columns = 'level_1')
    smod_half_cumu.to_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_half_cumu.csv')

