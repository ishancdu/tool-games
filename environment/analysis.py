import pandas as pd
import numpy as np
import pdb
from scipy.stats.stats import pearsonr as cor
import matplotlib.pyplot as plt



def read_data_model(model_result_path):
    """
    The function gives the output data for the model results
    """
    res_norm = pd.read_csv(model_result_path+'/game_stats_normal_gravity.csv')
    res_half = pd.read_csv(model_result_path+'/game_stats_half_gravity.csv')
    res_doub = pd.read_csv(model_result_path+'/game_stats_double_gravity.csv')

    return res_norm, res_half, res_doub

def read_data_part(part_path):
    
    part_norm = pd.read_csv(part_path+'/combined_res_norm_part.csv')
    part_half = pd.read_csv(part_path+'/combined_res_half_part.csv')
    part_doub = pd.read_csv(part_path+'/combined_res_double_part.csv')
    
    return part_norm, part_half, part_doub
    
if __name__ == '__main__':

    res_norm, res_half, res_doub = read_data_model(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs'
    )

    part_norm, part_half, part_doub = read_data_part(
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants'
    )

    pdb.set_trace()
    des_df = pd.DataFrame(columns = ['Stat', 'Gravity', 'Score type', 'Subject', 'Value'])

    #1
    mean_norm_ac_m = res_norm['accuracy'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'normal', 'Score type':'accuracy',
        'Subject':'model', 'Value': mean_norm_ac_m
    }
    #2
    mean_norm_at_m = res_norm['average_tries'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'normal', 'Score type':'average tries',
        'Subject':'model', 'Value': mean_norm_at_m
    }
    #3
    std_norm_ac_m = res_norm['accuracy'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'normal', 'Score type':'accuracy',
        'Subject':'model', 'Value': std_norm_ac_m
    }
    #4
    std_norm_at_m = res_norm['average_tries'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'normal', 'Score type':'average tries',
        'Subject':'model', 'Value': std_norm_at_m
    }
    #5
    mean_half_ac_m = res_half['accuracy'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'half', 'Score type':'accuracy',
        'Subject':'model', 'Value': mean_half_ac_m
    }
    #6
    mean_half_at_m = res_half['average_tries'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'half', 'Score type':'average tries',
        'Subject':'model', 'Value': mean_half_at_m
    }
    #7
    std_half_ac_m = res_half['accuracy'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'half', 'Score type':'accuracy',
        'Subject':'model', 'Value': std_half_ac_m
    }
    #8
    std_half_at_m = res_half['average_tries'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'half', 'Score type':'average tries',
        'Subject':'model', 'Value': std_half_at_m
    }
    #9
    mean_doub_ac_m = res_doub['accuracy'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'double', 'Score type':'accuracy',
        'Subject':'model', 'Value': mean_doub_ac_m
    }
    #10
    mean_doub_at_m = res_doub['average_tries'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'double', 'Score type':'average tries',
        'Subject':'model', 'Value': mean_doub_at_m
    }
    #11
    std_doub_ac_m = res_doub['accuracy'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'doub', 'Score type':'accuracy',
        'Subject':'model', 'Value': std_doub_ac_m
    }
    std_doub_at_m = res_doub['average_tries'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'double', 'Score type':'average tries',
        'Subject':'model', 'Value': std_doub_at_m
    }

    
    mean_norm_ac_p = part_norm['accuracy'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'normal', 'Score type':'accuracy',
        'Subject':'human', 'Value': mean_norm_ac_p
    }
    mean_norm_at_p = part_norm['average_tries'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'normal', 'Score type':'average tries',
        'Subject':'human', 'Value': mean_norm_at_p
    }
    std_norm_ac_p = part_norm['accuracy'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'normal', 'Score type':'accuracy',
        'Subject':'human', 'Value': std_norm_ac_p
    }
    std_norm_at_p = part_norm['average_tries'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'normal', 'Score type':'average tries',
        'Subject':'human', 'Value': std_norm_at_p
    }
    
    mean_half_ac_p = part_half['accuracy'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'half', 'Score type':'accuracy',
        'Subject':'human', 'Value': mean_half_ac_p
    }
    mean_half_at_p = part_half['average_tries'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'half', 'Score type':'average tries',
        'Subject':'human', 'Value': mean_half_at_p
    }
    std_half_ac_p = part_half['accuracy'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'half', 'Score type':'accuracy',
        'Subject':'human', 'Value': std_half_ac_p
    }
    std_half_at_p = part_half['average_tries'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'half', 'Score type':'average tries',
        'Subject':'human', 'Value': std_half_at_p
    }
    mean_doub_ac_p = part_doub['accuracy'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'double', 'Score type':'accuracy',
        'Subject':'human', 'Value': mean_doub_ac_p
    }
    mean_doub_at_p = part_doub['average_tries'].mean()
    des_df.loc[len(des_df)] = {
        'Stat': 'mean', 'Gravity':'double', 'Score type':'average tries',
        'Subject':'human', 'Value': mean_doub_at_p
    }
    std_doub_ac_p = part_doub['accuracy'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'doub', 'Score type':'accuracy',
        'Subject':'human', 'Value': std_doub_ac_p
    }
    std_doub_at_p = part_doub['average_tries'].std()
    des_df.loc[len(des_df)] = {
        'Stat': 'std', 'Gravity':'double', 'Score type':'average tries',
        'Subject':'human', 'Value': std_doub_at_p
    }

    des_df.to_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/descriptive_stat.csv', index=False)
    

    #between half and double gravity
    corr_hd_at = cor(res_half['average_tries'], res_doub['average_tries'])
    corr_hd_ac = cor(res_half['accuracy'], res_doub['accuracy'])

    fig, ax = plt.subplots()
    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for average tries is '+str(round(corr_hd_at.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_half['average_tries'], res_doub['average_tries'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("average attempt half gravity model", size=14)
    plt.ylabel("average attempt double gravity model", size=14)
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_doub['game_name']),title="game name")
    #plt.text(0.5, 0.5, 'correlation for average tries is'+str(round(corr_hd_at.statistic,3)), fontsize=14, bbox=dict(facecolor='red', alpha=0.5))
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/avg_tries_half_double.png')
    plt.clf()


    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for accuracy is '+str(round(corr_hd_ac.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_half['accuracy'], res_doub['accuracy'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("accuracy half gravity model", size=14)
    plt.ylabel("accuracy double gravity model", size=14)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_norm['game_name']),title="game name")
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/accuracy_half_double_model.png')
    plt.clf()
    

    #between norm and norm participant
    corr_norm_part_at = cor(res_norm['average_tries'], part_norm['average_tries'])
    corr_norm_part_ac = cor(res_norm['accuracy'], part_norm['accuracy'])

    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for average tries is '+str(round(corr_norm_part_at.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_norm['average_tries'], part_norm['average_tries'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("average attempt norm gravity model", size=14)
    plt.ylabel("average attempt norm gravity participant", size=14)
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_norm['game_name']),title="game name")
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/avg_tries_norm_gravity_mp.png')
    plt.clf()


    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for accuracy is '+str(round(corr_norm_part_ac.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_norm['accuracy'], part_norm['accuracy'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("accuracy norm gravity model", size=14)
    plt.ylabel("accuracy norm gravity participant", size=14)
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_norm['game_name']),title="game name")
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/accuracy_norm_gravity_mp.png')
    plt.clf()


    #between half part anf half model
    corr_half_part_at = cor(res_half['average_tries'], part_half['average_tries'])
    corr_half_part_ac = cor(res_half['accuracy'], part_half['accuracy'])

    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for average tries is '+str(round(corr_half_part_at.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_half['average_tries'], part_half['average_tries'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("average attempt half gravity model", size=14)
    plt.ylabel("average attempt half gravity participant", size=14)
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_half['game_name']),title="game name")
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/avg_tries_half_gravity_mp.png')
    plt.clf()


    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for accuracy is '+str(round(corr_half_part_ac.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_half['accuracy'], part_half['accuracy'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("accuracy half gravity model", size=14)
    plt.ylabel("accuracy half gravity participant", size=14)
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_half['game_name']),title="game name")
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/accuracy_half_gravity_mp.png')
    plt.clf()

    #between doub part and doub model
    corr_doub_part_at = cor(res_doub['average_tries'], part_doub['average_tries'])
    corr_doub_part_ac = cor(res_doub['accuracy'], part_doub['accuracy'])

    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for average tries is '+str(round(corr_doub_part_at.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_doub['average_tries'], part_doub['average_tries'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("average attempt double gravity model", size=14)
    plt.ylabel("average attempt double gravity participant", size=14)
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_doub['game_name']),title="game name")
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/avg_tries_doub_gravity_mp.png')
    plt.clf()


    plt.figure(figsize=(10,10))
    plt.text(0.5, 2.2, 'correlation for accuracy is '+str(round(corr_doub_part_ac.statistic,3)), fontsize=12, transform=ax.transAxes)
    scatter = plt.scatter(res_doub['accuracy'], part_doub['accuracy'], c=[1,2,3,4,5,6,7,8], s=100)
    plt.xlabel("accuracy double gravity model", size=14)
    plt.ylabel("accuracy double gravity participant", size=14)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(res_doub['game_name']),title="game name")
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.grid()
    plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/accuracy_doub_gravity_mp.png')
    plt.clf()

    


    #create the cumulative score graph

    #cumulative score graph for normal gravity

    mod_norm = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_norm_cumu.csv')
    part_norm = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants/cumulative_res_norm_part.csv')

    for game in mod_norm['game_name'].unique():
        mod = mod_norm[mod_norm['game_name']==game]
        part = part_norm[part_norm['game_name']==game]
        plt.step(mod['tries'], mod['accuracy'], linewidth = '8')
        plt.step(part['tries'], part['accuracy'], linewidth = '8')
        plt.ylabel("accuracy")
        plt.xlabel("attempt")
        plt.xlim(0, 12)
        plt.ylim(0, 120)
        plt.legend(['Model', 'Participant'])
        plt.title("Cumulative accuracy normal gravity for game "+ str(game))
        plt.grid()
        plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/cumulative_scores/norm_'+str(game)+'.png')
        plt.clf()



    mod_half = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_half_cumu.csv')
    part_half = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants/cumulative_res_half_part.csv')

    for game in mod_half['game_name'].unique():
        mod = mod_half[mod_half['game_name']==game]
        part = part_half[part_half['game_name']==game]
        plt.step(mod['tries'], mod['accuracy'], linewidth = '8')
        plt.step(part['tries'], part['accuracy'], linewidth = '8')
        plt.ylabel("accuracy")
        plt.xlabel("attempt")
        plt.xlim(0, 12)
        plt.ylim(0, 120)
        plt.legend(['Model', 'Participant'])
        plt.title("Cumulative accuracy half gravity for game "+ str(game))
        plt.grid()
        plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/cumulative_scores/half_'+str(game)+'.png')
        plt.clf()


    mod_doub = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_doub_cumu.csv')
    part_doub = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/participants/cumulative_res_doub_part.csv')

    for game in mod_doub['game_name'].unique():
        mod = mod_doub[mod_doub['game_name']==game]
        part = part_doub[part_doub['game_name']==game]
        plt.step(mod['tries'], mod['accuracy'], linewidth = '8')
        plt.step(part['tries'], part['accuracy'], linewidth = '8')
        plt.ylabel("accuracy")
        plt.xlabel("attempt")
        plt.xlim(0, 12)
        plt.ylim(0, 120)
        plt.legend(['Model', 'Participant'])
        plt.title("Cumulative accuracy double gravity for game "+ str(game))
        plt.grid()
        plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/cumulative_scores/double_'+str(game)+'.png')
        plt.clf()



    mod_doub = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_doub_cumu.csv')
    mod_half = pd.read_csv('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/model_cumu/model_half_cumu.csv')

    for game in mod_doub['game_name'].unique():
        mod = mod_doub[mod_doub['game_name']==game]
        part = mod_half[mod_half['game_name']==game]
        plt.step(mod['tries'], mod['accuracy'], linewidth = '8')
        plt.step(part['tries'], part['accuracy'], linewidth = '8')
        plt.ylabel("accuracy")
        plt.xlabel("attempt")
        plt.xlim(0, 12)
        plt.ylim(0, 120)
        plt.legend(['Model double', 'model half'])
        plt.title("Cumulative accuracy between double and half gravity for game "+ str(game))
        plt.grid()
        plt.savefig('/home/ishan/workspace/msc_dissertation/tool-games/environment/outputs/graphs/cumulative_scores/double_half_'+str(game)+'.png')
        plt.clf()
    
