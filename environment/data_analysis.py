#the file contains the script to analyse all the data
import os
import pdb

import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from random import randint
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr as cor
from scipy.stats import f_oneway
from scipy import stats

"""
######for double
if df['index'].iloc[i] == 35 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-70,10), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )

        elif df['index'].iloc[i] == 28 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-60,30), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 12 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-0,60), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        else:

###############all combines graph annotation djustment
    
        if df['index'].iloc[i] == 35 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(60,15), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )

        elif df['index'].iloc[i] == 34 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-80,35), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
            
        elif df['index'].iloc[i] == 15 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(10,70), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 12 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-20,10), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 33 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-20,10), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )

        elif df['index'].iloc[i] == 18 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-60,30), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 28 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(5,60), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 10 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-70,00), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 36 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-40,30), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 30 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-4,-60), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 8 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-50,-7), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 12 and not accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-5,70), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
"""


gravity_dict = {'normal':200, 'half':100, 'double':400}

def read_human_data(path):
    """
    The function reads the human participant data
    """
    
    data_file = os.listdir(path)
    df = pd.read_csv(path+'/'+data_file[0])
    df = df.rename(columns={"attempt #": 'total_attempt'})
    return df

def combine_model_data(path):
    """
    combines all the data into a single csv file
    """
    env_folders = os.listdir(path)
    comb_df = pd.DataFrame()
    
    for env in env_folders:
        games = os.listdir(path+'/'+env)
        for game in games:
            df = pd.read_csv(path+'/'+env+'/'+game)
            df['Gravity'] = gravity_dict[env]
            comb_df = pd.concat([comb_df, df])
    
    return comb_df

def desc_func(df):
    """
    The function gets the descriptive statistic for each game 
    """
    df_n = df.drop(df[(df["total_attempt"]>10)].index)
    attempt_mean = df_n.groupby(['participant ID']).agg({'total_attempt':max}).mean()
    accuracy = 100* (len(df_n[df_n['success']==1]['participant ID'].unique())\
                     /len(df['participant ID'].unique()))
    return pd.DataFrame({
        'accuracy_human': accuracy, 'average_attempt_human': attempt_mean[0]}, index=[0]
                        )


def desc_func_model(df):
    """
    The function gets the descriptive statistic for each game for model run
    """
    vals = df.groupby(['game iter']).agg({'total_attempt':max})
    vals['unsuccess'] = 0

    vals.loc[vals[vals['total_attempt']>10].index, 'unsuccess'] = 1
    vals.loc[vals[vals['total_attempt']>10].index, 'total_attempt'] = 10
    mean_trial = vals.mean()[0]
    accuracy = ((vals.shape[0]-vals['unsuccess'].sum())/vals.shape[0])*100

    return pd.DataFrame({
        'accuracy_model': accuracy, 'average_attempt_model': mean_trial}, index=[0]
                        )
    
    
def ssup_agg(df):
    
    vals = df.groupby(['Idx']).agg({'TotalAttempts':max})
    vals['unsuccess'] = 0
    vals.loc[vals[vals['TotalAttempts']>10].index, 'unsuccess'] = 1
    vals.loc[vals[vals['TotalAttempts']>10].index, 'TotalAttempts'] = 10
    mean_trial = vals.mean()[0]
    accuracy = ((vals.shape[0]-vals['unsuccess'].sum())/vals.shape[0])*100
    
    return pd.DataFrame({'accuracy': accuracy, 'average_attempt': mean_trial}, index=[0])


def plot_corr_chart(df, col1, col2, file_name, accuracy=False, gravity=True):
    """
    The function plots the correlation chart for the given 2 corrdinates
    """
    
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,10))
    plt.text(
        0.35, 2.3, 'correlation for '+  file_name.split('_')[0]+' is '\
        +str(round(cor(df[col1], df[col2]).statistic,3)),
        fontsize=20, transform=ax.transAxes
    )
    print (" the correlation between ", file_name, " is ", cor(df[col1], df[col2]),
           " and the degree of freedom is ", df.shape[0]-2)
    print (" the mean and std ", file_name, " for column", col1, " is ", df[col1].mean(), " ",
           df[col1].std())
    print (" the mean and std ", file_name, " for column", col2, " is ", df[col2].mean(), " ",
           df[col2].std())
           
    vals = stats.ttest_ind(df[col1], df[col2])
    print("studesnt t test score for ", col1, " and ", col2, "are ", vals)
    
    b, a = np.polyfit(df[col1], df[col2], deg=1)
    
    plt.plot(range(-10,120), range(-10,120), linestyle='dashed', color='black')
    plt.plot(range(-10,120), a+b*range(-10,120), linestyle='dashed', color='Red')
    #scatter = plt.scatter(df[col1], df[col2], c=range(1+45,len(df)+1+45), s=100)
    scatter = plt.scatter(df[col1], df[col2], s=100, color='black')
    #scatter = plt.scatter(df[col1], df[col2], s=100)
    
    plt.xlabel(col1, size=20)
    plt.ylabel(col2, size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    
    if accuracy:
        plt.xlim(-10, 120)
        plt.ylim(-10, 120)
        
    else:
        plt.xlim(-1, 12)
        plt.ylim(-1, 12)
        
    print(df)


    for i in range(len(df)):

        if df['index'].iloc[i] == 5 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(10,65), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        elif df['index'].iloc[i] == 8 and accuracy:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(40,65), textcoords='offset pixels',
                fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
            )
        else:
            plt.annotate(
                str(df['index'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(6,5), textcoords='offset pixels',
                fontsize=15
            )



            
    #texts = [plt.text(df[col1].iloc[i], df[col2].iloc[i],
    #                  str(df['index'].iloc[i]), ha='center', va='center') for i in range(
    #                      len(df))]
    #adjust_text(texts, fontsize=20, arrowprops=dict(arrowstyle='->', color='black', lw=.1))


    """
    

        if gravity:
            '''
            plt.annotate(
                df['game_name'].iloc[i]+"-"+str(df['Gravity'].iloc[i]),
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(6,5), textcoords='offset pixels'
            )
            
            if df['index'].iloc[i] == 35:
                plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(60,15), textcoords='offset pixels', fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
                )

            elif df['index'].iloc[i] == 34:
                plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-80,35), textcoords='offset pixels', fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
                )
                
                
            else:
                plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(randint(-160, 160),randint(-160, 160)), textcoords='offset pixels', fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
                )
            '''
            
            plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(10,7), textcoords='offset pixels', fontsize=15
                )
        else:
            '''
            plt.annotate(
                df['game_name'].iloc[i],
                (df[col1].iloc[i], df[col2].iloc[i]), xytext=(6,5), textcoords='offset pixels'
            )

            if df['index'].iloc[i] == 35:
                plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(60,15), textcoords='offset pixels', fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
                )
            elif df['index'].iloc[i] == 34:
                plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(-80,35), textcoords='offset pixels', fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
                )
                
            else:
                plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(randint(-160, 160),randint(-160, 160)), textcoords='offset pixels', fontsize=15, arrowprops = dict(arrowstyle='->', lw=1, color='black')
                )
            '''
            plt.annotate(
                    str(df['index'].iloc[i]),
                    (df[col1].iloc[i], df[col2].iloc[i]), xytext=(10,7), textcoords='offset pixels', fontsize=15
                )
    '''
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(df['game_name']+ "-"+ df['Gravity'].astype('str')),
        title="game name Gravity"
    )
    '''
    """
    plt.grid()

    plt.savefig(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/'+file_name
    )
    
    plt.clf()



def calc_cumulate_hum(df):
    """
    The function calculates the cumulative scores for participant data 
    """

    total_part = len(df['participant ID'].unique())
    df_cumu = pd.DataFrame(columns = ['attempt', 'accuracy'])
    for trn in range(1,11):
        accuracy = (len(df[(df['success']==1) & (df['total_attempt']<=trn)][
            'participant ID'].unique())/total_part)*100
        df_cumu.loc[len(df_cumu)] = {'attempt':trn, 'accuracy':accuracy}
        #df_cumu = pd.concat(
        #    [df_cumu, pd.DataFrame({'tries':trn, 'accuracy':accuracy}, index=)],
        #    ignore_index=True)
        
    return df_cumu


def calc_cumulate_mod(df):
    """
    Calculate the cumulative data for model
    """
    total_part = len(df['game iter'].unique())
    df_cumu = pd.DataFrame(columns = ['attempt', 'accuracy'])
    
    for trn in range(1,11):
        accuracy = (len(df[(df['success']==True) & (df['total_attempt']<=trn)][
            'game iter'].unique())/total_part)*100
        df_cumu.loc[len(df_cumu)] = {'attempt':trn, 'accuracy':accuracy}
        #df_cumu = pd.concat(
        #    [df_cumu, pd.DataFrame({'tries':trn, 'accuracy':accuracy}, index=)],
        #    ignore_index=True)

    return df_cumu



def cumu_graph(df,gravity, level):
    """
    The function generates the cumulative graphs
    """
    plt.figure(figsize=(10,10))
    sample = df[(df['game_name']==level) & (df['Gravity']==gravity_dict[gravity])]
    if len(sample) == 0:
        return None
    
    plt.step(sample['attempt'], sample['model_accuracy'], linewidth = '8')
    plt.step(sample['attempt'], sample['human_accuracy'], linewidth = '8')
    plt.ylabel("accuracy", fontsize=30)
    plt.xlabel("attempt", fontsize=30)
    plt.xlim(0, 12)
    plt.xticks(fontsize=30)
    plt.ylim(0, 120)
    plt.yticks(fontsize=30)
    plt.legend(['Model', 'Participant'], fontsize=30)
    
    plt.title(level +"," \
              + gravity+" gravity ", fontsize=30)
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/output_graphs_main/cumulative_charts/'\
                +str(level)+"-"+gravity+'.png')
    plt.clf()

def t_test_func(df, col1, col2):
    
    vals = stats.ttest_ind(df[col1], df[col2])
    print("studesnt t test score for ", col1, " and ", col2, "are ", vals)
    return vals


def calc_model_v_human(df, cols1, cols2):
    """
    Creates the graph between model vs humans
    """

def bar_chart(df, accuracy=True):
    """
    The function draws the bar chats for the
    """

    bars = ['half', 'normal', 'double']
    fig, ax = plt.subplots()
    if accuracy:
        ax.bar(
            bars, 
            [df['accuracy_model-half'].mean(),
             df['accuracy_model-normal'].mean(),
             df['accuracy_model-double'].mean()
             ], yerr=[
                 df['accuracy_model-half'].std(),
                 df['accuracy_model-normal'].std(),
                 df['accuracy_model-double'].std()],
            align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Accuracy Model', fontsize=20)
    else:
        ax.bar(
            bars,
            [df['average_attempt_model-half'].mean(),
             df['average_attempt_model-normal'].mean(),
             df['average_attempt_model-double'].mean()
             ], yerr=[
                 df['average_attempt_model-half'].std(),
                 df['average_attempt_model-normal'].std(),
                 df['average_attempt_model-double'].std()],
            align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Average attempt Model', fontsize=20)
        
    ax.set_xlabel('Gravity', fontsize=20)
    plt.xticks(fontsize=20)
    plt.xticks(fontsize=20)
    #ax.xticks(fontsize=20)
    #ax.yticks(fontsize=20)
    #ax.set_xticks('Gravity')
    #ax.set_xticklabels(materials)
    #ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    if accuracy:
        plt.savefig('outputs/bar_plot_with_error_bars_accuracy.png')
    else:
        plt.savefig('outputs/bar_plot_with_error_bars_attempt.png')

    
if __name__ == '__main__':

    #original ssup model data 
    ssup_original_data \
        = '/home/ishan/workspace/msc_dissertation/tool-games/data/original_levels/models/ssup_full.csv'
    ssup_val_data \
        = '/home/ishan/workspace/msc_dissertation/tool-games/data/cv_levels/models/ssup_full.csv'
    
    ss_df = pd.read_csv(ssup_original_data)
    ss_stat = ss_df.groupby('Trial').apply(ssup_agg)
    ss_stat = ss_stat.reset_index().drop(columns = ['level_1'])
    ss_stat = ss_stat.rename(columns = {'Trial':'game_name'})

    sv_df = pd.read_csv(ssup_val_data)
    sv_stat = sv_df.groupby('Trial').apply(ssup_agg)
    sv_stat = sv_stat.reset_index().drop(columns = ['level_1'])
    sv_stat = sv_stat.rename(columns = {'Trial':'game_name'})
    



    
    data_output_path \
        = '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_output'
    human_data_path \
        = '/home/ishan/workspace/msc_dissertation/tool-games/environment/data'

    
    def func_part(df_p):
        return pd.DataFrame(
            columns = ['no_of_levels', 'gravity types'],
            data = [[df_p['game_name'].unique().shape[0],
             df_p['Gravity'].unique().shape[0]]]
                            )

    #combined human df 
    human_df = read_human_data(human_data_path)
    human_df = human_df.drop(human_df[human_df['game_name'].isin(['Funnel', 'FridgeMagnet'])].index)
    #combined model df
    vals_part = human_df.groupby('participant ID').apply(func_part)
    
    model_df = combine_model_data(data_output_path)

    model_cumu = model_df.groupby(['game_name', 'Gravity']).apply(calc_cumulate_mod)
    model_cumu = model_cumu.reset_index().drop(columns = ['level_2'])

    human_cumu = human_df.groupby(['game_name', 'Gravity']).apply(calc_cumulate_hum)
    human_cumu = human_cumu.reset_index().drop(columns = ['level_2'])
    
    model_cumu = model_cumu.rename(columns = {"accuracy":"model_accuracy"})
    human_cumu = human_cumu.rename(columns = {"accuracy":"human_accuracy"})

    comb_cumu = pd.merge(human_cumu, model_cumu, on=['game_name','Gravity','attempt'], how='left')

    '''
    for lev in comb_cumu['game_name'].unique():
        for gvt in ['half', 'normal', 'double']:
            cumu_graph(comb_cumu, gravity=gvt, level=lev)
    '''

    #use this code find the solution rate between trials to
    #find out if theres is any learning in participants






    #get descriptive statistics game wise for participants
    desc_human_df = human_df.groupby(['game_name', 'Gravity']).apply(desc_func)
    desc_human_df = desc_human_df.reset_index().drop(columns=['level_2'])

    #get descriptive statistics game wise for model
    desc_model_df = model_df.groupby(['game_name', 'Gravity']).apply(desc_func_model)
    desc_model_df = desc_model_df.reset_index().drop(columns = ['level_2'])

    ppr_df = pd.concat([ss_stat,sv_stat])
    nu_graph = pd.merge(
        ppr_df, desc_model_df[desc_model_df['Gravity']==200], on='game_name', how='inner'
    )
    ssup_comp = pd.merge(
        ss_stat, desc_model_df[desc_model_df['Gravity']==200], on='game_name', how='inner'
    ).reset_index()
    nu_graph = nu_graph.reset_index()
    ssup_comp = ssup_comp.rename(columns = {
        'average_attempt':'original ssup average attempt', 'accuracy':'original ssup accuracy'
    })
    
    plot_corr_chart(
        ssup_comp, 'original ssup average attempt', 'average_attempt_model',
        file_name='ssup_orig.png', gravity=False, accuracy=False
    )
    plot_corr_chart(
        ssup_comp, 'original ssup accuracy', 'accuracy_model',
        file_name='acc_ssup_orig.png', gravity=False, accuracy=True
    )
    print(ssup_comp[['index', 'game_name']])
    pdb.set_trace()
    '''
    plot_corr_chart(
        nu_graph, 'average_attempt', 'average_attempt_model',
        file_name='attepmt_orig.png', gravity=False
    )
    
    plot_corr_chart(
        nu_graph, 'accuracy', 'accuracy_model',
        file_name='accuracy_orig.png', gravity=False, accuracy=True
    )
    '''
    
    df_comb_stat = pd.merge(desc_human_df, desc_model_df, on=['game_name', 'Gravity'], how='inner')

    #df_comb_stat.loc[df_comb_stat[df_comb_stat['Gravity']==100].index, 'Gravity'] = 'half'
    #df_comb_stat.loc[df_comb_stat[df_comb_stat['Gravity']==200].index, 'Gravity'] = 'normal'
    #df_comb_stat.loc[df_comb_stat[df_comb_stat['Gravity']==400].index, 'Gravity'] = 'double'
    
    #df_comb_stat['level'] = df_comb_stat['game_name'] + '-' + df_comb_stat['Gravity']
    df_comb_stat = df_comb_stat.reset_index()
    df_comb_stat['index'] = df_comb_stat['index']+1
    #comparison between original model and replicated model
    df_comp = pd.merge(
        ss_stat, desc_model_df[desc_model_df['Gravity']==200], on='game_name', how='left'
    )
    df_val_comp = pd.merge(
        sv_stat, desc_model_df[desc_model_df['Gravity']==200], on='game_name', how='inner'
    )

    #include in the paper to get the correlation after removing the values
    #df_comb_stat = df_comb_stat.drop([7,20])
    
    df_comb_stat.to_csv("outputs/data_comb.csv")


    '''
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 200].drop([20,7]), 'average_attempt_human',
        'average_attempt_model',
        file_name='alt_attempt_200_human_model.png', gravity=False
    )        
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 200].drop([20,7]), 'accuracy_human', 'accuracy_model',
        file_name='alt_accuracy_200_human_model.png', gravity=False, accuracy=True
    )



    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 400], 'average_attempt_human', 'average_attempt_model',
        file_name='attempt_400_human_model.png', gravity=False
    )        
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 400], 'accuracy_human', 'accuracy_model',
        file_name='accuracy_400_human_model.png', gravity=False, accuracy=True
    )
    '''
    '''
    plot_corr_chart(
        df_comb_stat, 'accuracy_human', 'accuracy_model',
        file_name = 'accuracy_human_model.png', accuracy=True
    )
    
    plot_corr_chart(
        df_comb_stat, 'average_attempt_human', 'average_attempt_model',
        file_name='attempt_human_model.png'
    )
    '''

    '''
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 200], 'average_attempt_human', 'average_attempt_model',
        file_name='attempt_200_human_model.png', gravity=False
    )        
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 200], 'accuracy_human', 'accuracy_model',
        file_name='accuracy_200_human_model.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 100], 'average_attempt_human', 'average_attempt_model',
        file_name='attempt_100_human_model.png', gravity=False
    )
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 100], 'accuracy_human', 'accuracy_model',
        file_name='accuracy_100_human_model.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 400], 'average_attempt_human', 'average_attempt_model',
        file_name='attempt_400_human_model.png', gravity=False
    )
    plot_corr_chart(
        df_comb_stat[df_comb_stat['Gravity'] == 400], 'accuracy_human', 'accuracy_model',
        file_name='accuracy_400_human_model.png', gravity=False, accuracy=True
    )

    '''


    
    pvt_model = desc_model_df.pivot(
        index='game_name', columns='Gravity', values=['accuracy_model', 'average_attempt_model']
    )
    pvt_model.columns = [col[0]+'-'+str(col[1]) for col in pvt_model.columns]
    pvt_model = pvt_model.reset_index()
    pvt_model = pvt_model.reset_index()
    pvt_model['index'] = pvt_model['index']+1
    #print(pvt_model[['index', 'game_name']])
    pvt_model = pvt_model.rename(columns = {
        'accuracy_model-100':'accuracy_model-half', 'accuracy_model-200':'accuracy_model-normal',
        'accuracy_model-400':'accuracy_model-double',
        'average_attempt_model-100':'average_attempt_model-half',
        'average_attempt_model-200':'average_attempt_model-normal',
        'average_attempt_model-400':'average_attempt_model-double', })
    
    #bar_chart(pvt_model)
    #bar_chart(pvt_model, accuracy=False)

    '''
    #In this section we are gonna create the graphs for the model vs model analysis
    plot_corr_chart(
        pvt_model, 'accuracy_model-half', 'accuracy_model-double',
        file_name='accuracy_100_400_model.png', gravity=False, accuracy=True
    )

    plot_corr_chart(
        pvt_model, 'average_attempt_model-half', 'average_attempt_model-double',
        file_name='at_100_400_model.png', gravity=False, accuracy=False
    )

    plot_corr_chart(
        pvt_model, 'accuracy_model-half', 'accuracy_model-normal',
        file_name='accuracy_100_200_model.png', gravity=False, accuracy=True
    )

    plot_corr_chart(
        pvt_model, 'average_attempt_model-half', 'average_attempt_model-normal',
        file_name='at_100_200_model.png', gravity=False, accuracy=False
    )
    plot_corr_chart(
        pvt_model, 'accuracy_model-normal', 'accuracy_model-double',
        file_name='accuracy_200_400_model.png', gravity=False, accuracy=True
    )

    plot_corr_chart(
        pvt_model, 'average_attempt_model-normal', 'average_attempt_model-double',
        file_name='at_200_400_model.png', gravity=False, accuracy=False
    )
    '''
    
    '''
    plot_corr_chart(
        pvt_model, 'accuracy_model-normal', 'accuracy_model-half',
        file_name='accuracy_200_100_model.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        pvt_model, 'average_attempt_model-half', 'average_attempt_model-double',
        file_name='average_attempt_model_100_400_part.png', gravity=False, accuracy=False
    )
    plot_corr_chart(
        pvt_model, 'average_attempt_model-normal', 'average_attempt_model-double',
        file_name='average_attempt_model_200_400_model.png', gravity=False, accuracy=False
    )
    plot_corr_chart(
        pvt_model, 'average_attempt_model-normal', 'average_attempt_model-half',
        file_name='average_attempt_model_200_100_model.png', gravity=False, accuracy=False
    )

    '''

    pvt_human = desc_human_df.pivot(
        index='game_name', columns='Gravity', values=['accuracy_human', 'average_attempt_human']
    )
    pvt_human.columns = [col[0]+'-'+str(col[1]) for col in pvt_human.columns]
    pvt_human = pvt_human.reset_index()

    acc_pvt_hum = pvt_human[['game_name', 'accuracy_human-100', 'accuracy_human-400', 'average_attempt_human-100', 'average_attempt_human-400']].dropna()
    acc_pvt_hum = acc_pvt_hum.rename(columns = {
        'accuracy_human-100':'accuracy_human-half', 'accuracy_human-400':'accuracy_human-double',
        'average_attempt_human-100':'average_attempt_human-half', 'average_attempt_human-400':'average_attempt_human-double'}
                       )
    acc_pvt_mod = pvt_model[pvt_model['game_name'].isin(acc_pvt_hum['game_name'].unique())]
    acc_pvt_hum = acc_pvt_hum.reset_index().drop(columns = 'index').reset_index()
    acc_pvt_hum['index'] = acc_pvt_hum['index']+1
    print(acc_pvt_hum[['index', 'game_name']])
    acc_pvt_mod['index'] = acc_pvt_hum['index'].values
    
    plot_corr_chart(
        acc_pvt_hum, 'accuracy_human-half', 'accuracy_human-double',
        file_name='acc_100_400_hum.png', gravity=False, accuracy=True
    )
    
    plot_corr_chart(
        acc_pvt_mod, 'accuracy_model-half', 'accuracy_model-double',
        file_name='acc_100_400_mod.png', gravity=False, accuracy=True
    )

    '''
    t_test_func(pvt_model, 'accuracy_model-half', 'accuracy_model-double')
    t_test_func(pvt_model, 'accuracy_model-half', 'accuracy_model-normal')
    t_test_func(pvt_model, 'accuracy_model-normal', 'accuracy_model-double')
    t_test_func(pvt_model, 'average_attempt_model-half', 'average_attempt_model-double')
    t_test_func(pvt_model, 'average_attempt_model-half', 'average_attempt_model-normal')
    t_test_func(pvt_model, 'average_attempt_model-normal', 'average_attempt_model-double')
    '''

    '''
    plot_corr_chart(
        acc_pvt_mod, 'average_attempt_model-half', 'average_attempt_model-double',
        file_name='aat_100_400_part.png', gravity=False, accuracy=False
    )
    '''
    #plot_corr_chart(
    #    acc_pvt_mod, 'average_attempt_model-half', 'average_attempt_model-double',
    #    file_name='average_attempt_100_400_part.png', gravity=False, accuracy=False
    #)
    
    pdb.set_trace()


    '''
    plot_corr_chart(
        acc_pvt_hum, 'accuracy_human-half', 'accuracy_human-double',
        file_name='accuracy_100_400_part.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        acc_pvt_hum, 'average_attempt_human-half', 'average_attempt_human-double',
        file_name='average_attempt_100_400_part.png', gravity=False, accuracy=False
    )
    '''
    '''
    plot_corr_chart(
        pvt_model, 'accuracy_model-100', 'accuracy_model-400',
        file_name='accuracy_100_400_model.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        pvt_model, 'accuracy_model-200', 'accuracy_model-400',
        file_name='accuracy_200_400_model.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        pvt_model, 'accuracy_model-200', 'accuracy_model-100',
        file_name='accuracy_200_100_model.png', gravity=False, accuracy=True
    )
    plot_corr_chart(
        pvt_model, 'average_attempt_model-200', 'average_attempt_model-100',
        file_name='average_attempt_200_100_model.png', gravity=False, accuracy=False
    )
    plot_corr_chart(
        pvt_model, 'average_attempt_model-200', 'average_attempt_model-400',
        file_name='average_attempt_200_400_model.png', gravity=False, accuracy=False
    )
    plot_corr_chart(
        pvt_model, 'average_attempt_model-100', 'average_attempt_model-400',
        file_name='average_attempt_100_400_model.png', gravity=False, accuracy=False
    )
    '''

    
    

