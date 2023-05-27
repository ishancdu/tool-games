import json
import os
import pdb

import pygame as pg

from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement
from pyGameWorld import noisifyWorld
from pyGameWorld import ssup_model
import pandas as pd

Game_list = [
    'Gap.json', 'Shafts_B.json', 'Table_A.json', 
    'Launch_B.json', 'Table_B.json', 'Unbox.json', 
    'Falling_A.json', 'Falling_B.json', 'Bridge.json', 
    'Launch_A.json', 'Basic.json', 'Catapult.json', 
    'Prevention_A.json', 'Prevention_B.json', 'Unsupport.json', 
    'SeeSaw.json', 'Chaining.json', 'Shafts_A.json'
    ]

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

if __name__ == '__main__':

    '''
    game_obj = read_the_game_file(
        'Catapult.json',
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/Original/'
        )
    

    '''
    game_name = 'Catapult.json'
    
    game_obj = read_the_game_file(
        game_name,
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/Original/'
        )
    
    main_out = pd.DataFrame()
    for count in range(25):
        success, out_df = ssup_model.SSUP_model_run(game_obj, game_name.split('.')[0], count)
        out_df['success'] = success
        out_df['total_attempt'] = out_df.shape[0]
        main_out = pd.concat([main_out, out_df])
        main_out.to_csv("output_"+game_name.split('.')[0]+".csv", index=False)

    # Find the path of objects over 2s
    # Comes out as a dict with the moveable object names
    # (PLACED for the placed tool) with a list of positions over time each
    #path_dict, success, time_to_success = game_obj.observePlacementPath(toolname="obj2",position=(300,400),maxtime=20.)


    pdb.set_trace()
