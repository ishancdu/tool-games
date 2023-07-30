import json
import os
import pdb

import pygame as pg

from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement
from pyGameWorld import noisifyWorld
from pyGameWorld import ssup_model
import pandas as pd

gravity = 'normal'

working_games = [
    'Gap.json', 'Unbox.json', 'Falling_A.json', 'Basic.json',
    'Prevention_A.json', 'Prevention_A.json', 'Basic_v2.json',
    'Launch_v2.json', 'Spiky.json', 'Trap.json', 'Shafts_B.json',
    'Shafts_A.json' ]

na = [
    'BridgeAlt', 'BalanceUnder', 'Backup', 'Filler', 'FridgeMagnet',
    'Funnel', 'DoubleDown'
]



not_in_regulat = ['BridgeAlt', 'BalanceUnder', 'Backup', 'Filler']


not_in_alg = ['Towers_B', 'FridgeMagnet' ,'Catapult_Alt','Funnel' ,'Towers_A',
              'DoubleDown']


def read_the_game_file(file_name, path):
    """
    The function reads the game parameter file 
    and loads the data into the game objects
    path(str) --> path for the file
    file_name(str) --> name of the game
    """
    read_data = open(path+file_name).read()
    dict_game = json.loads(read_data)
    tp = ToolPicker(dict_game)

    return tp

total_iter = 50

def run_games(game_dict):
    """
    Runs the games and saves the data in the data folder
    """
    for game in game_dict['games']:

        #read the game file and create the game obj
        '''
        game_obj = read_the_game_file(
            game,
            '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/'
        )
        '''
        print("Running the game --> ", game)
        game_obj = read_the_game_file(
            game,
            game_dict['path']
        )
        
        if game_dict['gravity'] == 'double':
            game_obj._worlddict['gravity'] = game_obj._worlddict['gravity']*2
        
        elif game_dict['gravity'] == 'half':
            game_obj._worlddict['gravity'] = game_obj._worlddict['gravity']/2

        
        main_out = pd.DataFrame()
        for count in range(total_iter):
            
            success, out_df = ssup_model.SSUP_model_run(game_obj, game.split('.')[0], count)
            out_df['success'] = success
            out_df['total_attempt'] = out_df.shape[0]
            main_out = pd.concat([main_out, out_df])
            main_out.to_csv("outputs/double_new/"+game.split('.')[0]+".csv", index=False)
            
        
if __name__ == '__main__':


    '''
    game_dict_normal = {
        'gravity': 'normal',
        'games': [
            'Falling_B.json', 'Falling_v2.json', 'Shove.json',
            'Balance.json', 'Table_v2.json', 'Launch_B.json',
            'Table_B.json', 'Unsupport.json'
        ]
    }
    game_dict_unp = {
        'gravity': 'normal',
        'games': [
            'Unsupport.json'
        ]
    }
    game_dict_half = {
        'gravity': 'half',
        'games': [
            'Catapult.json', 'Launch_A.json', 'Chaining.json',
            'Bridge.json', 'Table_A.json', 'Remove.json', 'Collapse.json',
            'SeeSaw.json'
        ]
    }

    game_dict_cs_half = {
        'gravity': 'half',
        'games': [
            'Collapse.json',
            'SeeSaw.json'
        ]
    }
    
    game_dict_doub = {
        'gravity': 'double',
        'games': [
            'Catapult.json', 'Launch_A.json', 'Chaining.json',
            'Bridge.json', 'Table_A.json', 'Remove.json', 'Collapse.json',
            'SeeSaw.json'
        ]
    }
    
    game_dict_new_half = {
        'gravity': 'double',
        'path' : '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/new/', 
        'games': os.listdir('/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/new/')
        
    }

    game_dict_done_normal_half = {
        'gravity': 'double',
        'path' : '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/done_normal/', 
        'games': os.listdir('/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/done_normal/')
        
    }
    '''
    game_dict_new_normal = {
        'gravity': 'double',
        'path' : '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/rem_games/', 
        'games': os.listdir('/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/all_games/rem_games/')
        
    }
    


    
    #run_games(game_dict_new_half)
    run_games(game_dict_new_normal)
    # Find the path of objects over 2s
    # Comes out as a dict with the moveable object names
    # (PLACED for the placed tool) with a list of positions over time each
    #path_dict, success, time_to_success = game_obj.observePlacementPath(toolname="obj2",position=(300,400),maxtime=20.)


