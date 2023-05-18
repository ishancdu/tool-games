import json
import os
import pdb

import pygame as pg

from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement
from pyGameWorld import noisifyWorld
from pyGameWorld import ssup_model

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

    
    game_obj = read_the_game_file(
        'Catapult.json',
        '/home/ishan/workspace/msc_dissertation/tool-games/environment/Trials/Original/'
        )
    
    ssup_model.SSUP_model_run(game_obj)
    
    # Find the path of objects over 2s
    # Comes out as a dict with the moveable object names
    # (PLACED for the placed tool) with a list of positions over time each
    #path_dict, success, time_to_success = game_obj.observePlacementPath(toolname="obj2",position=(300,400),maxtime=20.)


    pdb.set_trace()
