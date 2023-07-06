#gets the correlation between participants

from scipy.stats.stats import pearsonr as cor
import matplotlib.pyplot as plt
import pandas as pd
import pdb


def calc_stat(df):
    accuracy = (len(df[(df['success']==1) & (df['attempt #']<=10)][
        "participant ID"
    ].unique())/len(df[
        "participant ID"].unique()))*100
    
    df_d = df.drop(df[(df["attempt #"]>10)].index)
    tries = df_d["attempt #"].mean()
    
    ddf = pd.DataFrame({'accuracy':accuracy, 'average_tries':tries}, index=[0])
    return ddf



if __name__ == '__main__':


    df = pd.read_csv(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/data/HumanDataForComparison.csv'
    )

    half = [
        'Bridge', 'Catapult', 'Chaining', 'Collapse', 'Launch_A', 'Remove', 'SeeSaw', 'Table_A'
    ]

    df_half = df[(df['Gravity']==100) & df['game_name'].isin(half)]
    
    stats_half = df_half.groupby('game_name').apply(calc_stat).reset_index()
    stats_half = stats_half.drop(columns = 'level_1')


    df_doub = df[(df['Gravity']==400) & df['game_name'].isin(half)]
    
    stats_doub = df_doub.groupby('game_name').apply(calc_stat).reset_index()
    stats_doub = stats_doub.drop(columns = 'level_1')



    corr_hd_at = cor(stats_half['average_tries'], stats_doub['average_tries'])
    corr_hd_ac = cor(stats_half['accuracy'], stats_doub['accuracy'])

    fig, ax = plt.subplots()
    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for average attempts is '+str(round(corr_hd_ac.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(stats_half['average_tries'], stats_doub['average_tries'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("average attempts half gravity participant", size=14)
    plt.ylabel("average attempts double gravity participant", size=14)
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(stats_doub['game_name']),
               title="game name")
    #plt.text(0.5, 0.5, 'correlation for average tries is'+str(round(corr_hd_at.statistic,3)), fontsize=14, bbox=dict(facecolor='red', alpha=0.5))
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/avg_tries_half_double_participant.png')
    plt.clf()



    fig, ax = plt.subplots()
    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for accuracy is '+str(round(corr_hd_at.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(stats_half['accuracy'], stats_doub['accuracy'],
                          c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("accuracy half gravity participant", size=14)
    plt.ylabel("accuracy double gravity participant", size=14)
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(stats_doub['game_name']),
               title="game name")
    #plt.text(0.5, 0.5, 'correlation for average tries is'+str(round(corr_hd_at.statistic,3)), fontsize=14, bbox=dict(facecolor='red', alpha=0.5))
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/accuracy_half_double_participant.png')
    plt.clf()

    pdb.set_trace()
