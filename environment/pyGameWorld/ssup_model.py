#the file contains the code for the ssup mdoels

import random
import pdb

import numpy as np


def calc_reward(world_act, init_world):
    """
    Normalized distance between target and reward
    args:
    world_act(dict) --> positions after the actions is performed
    init_world(dict) --> initial world position
    """
    #intial position
    init_ball = init_world['objects']['Ball']['position']
    x_init_g = (
    init_world['objects']['Goal']['points'][0][0] \
    + init_world['objects']['Goal']['points'][2][0] \
    + init_world['objects']['Goal']['points'][1][0] \
    + init_world['objects']['Goal']['points'][3][0]) \
    /4
    y_init_g = (
    init_world['objects']['Goal']['points'][0][1] \
    + init_world['objects']['Goal']['points'][1][1] \
    + init_world['objects']['Goal']['points'][2][1] \
    + init_world['objects']['Goal']['points'][3][1]) \
    /4
    init_goal = [x_init_g, y_init_g]
    init_dist = np.sqrt((x_init_g-init_ball[0])**2 + (y_init_g-init_ball[1])**2)

    #final positionGoal 
    final_ball = world_act['objects']['Ball']['position']
    x_fin_g = (
    world_act['objects']['Goal']['points'][0][0] \
    + world_act['objects']['Goal']['points'][1][0] \
    + world_act['objects']['Goal']['points'][2][0] \
    + world_act['objects']['Goal']['points'][3][0]) \
    /4
    y_fin_g = (
    world_act['objects']['Goal']['points'][0][1] \
    + world_act['objects']['Goal']['points'][1][1] \
    + world_act['objects']['Goal']['points'][2][1] \
    + world_act['objects']['Goal']['points'][3][1]) \
    /4
    final_goal = [x_fin_g, y_fin_g]
    final_dist = np.sqrt((x_fin_g-final_ball[0])**2 + (y_fin_g-final_ball[1])**2)
    return (1-(final_dist/init_dist))

def init_policy():
    """
    The function initializes the policy of the given 
    enviornment
    """


def simulate(game_obj, init_pts):
    """
    Run simulation based on initial points to get noisy rewards
    init_pts(dict) --> initial points for each tool
    """
    tool = random.sample(init_pts.keys(), 1)[0]
    
    position = init_pts[tool]
    noise_dict = {
        'noise_position_static': 5., 'noise_position_moving': 5.,
        'noise_collision_direction': .2, 'noise_collision_elasticity': .2, 'noise_gravity': .1,
        'noise_object_friction': .1, 'noise_object_density': .1, 'noise_object_elasticity': .1
                 }
    path_dict, success, time_taken, world_act = game_obj.runFullNoisyPath(tool, position, maxtime=20., returnDict=True, **noise_dict)

    reward = calc_reward(world_act, game_obj._worlddict)
    
    return reward



def sample(world_dict, tools, y_dist=200, x_dist=20):
    """
    Samples the initial points based on the object oriented priors
    world_dict(dict): movable objects coordinates that is used to sample 
    the initial points
    inti_points(dict): intitialized points
    """
    #get dynamic objects y mean and x range
    dynamic_obj = {}
    for key, value in world_dict.items():
        if value['color'] in ['red', 'blue']:
            if 'position' in value.keys():
                y_mean = value['position'][1]
                x_range = [
                    value['position'][0]-value['radius'], 
                    value['position'][0]+value['radius']
                    ]
                value.update({'y_mean':y_mean, 'x_range':x_range})
                dynamic_obj.update({key:value})

            elif 'polys' in value.keys():
                x_max = 0
                x_min = 600
                y_sum = 0
                y_cnt = 0
                for row in value['polys']:
                    for col in row:
                        if col[0] > x_max:
                            x_max = col[0]
                        if col[0] < x_min:
                            x_min = col[0]
                        y_sum = y_sum + col[1]
                        y_cnt = y_cnt + 1
                
                y_mean = int(y_sum/y_cnt)        
                value.update({'y_mean':y_mean, 'x_range':[x_min, x_max]})
                dynamic_obj.update({key:value})
            else:
                print("new structure for the dynamic object")
                pdb.set_trace()
    

    #sample points
    init_pts = {} 
    #sample an initial point for each tool
    for tool in tools.keys():
        #randomly sample a dynamic object
        samp_dyn_obj = random.sample(dynamic_obj.keys(),1)[0]
        
        #sample y points basedo n gaussian dist
        y_init = int(np.random.normal(dynamic_obj[samp_dyn_obj]['y_mean'], y_dist,1))

        #sample x uniformly
        x_0 = dynamic_obj[samp_dyn_obj]['x_range'][0]-x_dist
        x_1 = dynamic_obj[samp_dyn_obj]['x_range'][1]+x_dist
        x_init = random.sample(range(x_0, x_1), 1)[0]
        
        init_pts.update({tool:[x_init, y_init]})

    return init_pts
        

    

def SSUP_model_run(world):
    """
    The function runs the SSUP model
    (Sample, Simulate and Update model)
    Args:
    world(dict): Contains the enviornment data values
    Return:
    model_run(Pandas df): Pandas dataframe for the model run values
    best_pos(corrdinated): best coordinate values for the given enviornment
    """
    #initialized points for each tool
    init = sample(world._worlddict['objects'], world._tools)
    
    #simulate the actions
    rewards = simulate(world, init)


    pdb.set_trace()

