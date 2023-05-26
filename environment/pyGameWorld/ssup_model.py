#the file contains the code for the ssup mdoels

import random
import math
import pdb
import time

import numpy as np
import pandas as pd
from scipy.stats import norm as gaussian_dist
from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement

#parameters for the ssup model
eps = 0.30 #value for the greedy strategy
init_pts = 3 #initial points for which the policy is inititialized
sims = 4 #number of simulation that are being run for which the average reward is calculated
iters = 5 #maximum number of simulation after which the action needs to be performed
thresh = 0.5 #reward threshold for acting
lr = .1 #learning rate for policy gradient
scl = 600

def exploration():
    if random.random()<0.3:
        return True
    else:
        return False

def sample(world, dynamic_obj, tools, tool_points_no=1, y_dist=200, x_dist=20):
    """
    Samples the points based on the object oriented priors

    Args:
    dynamic_obj(dict) -> objects dictionary coordinates that is used to sample 
    the initial points
    tools(dict)-> Tools being used in the enviornment

    Return
    init_points(dict): initialized points
    """
    
    #sample points
    init_pts = {}

    #sample an initial point for each tool
    for tool in tools.keys():
        for t_pts in range(tool_points_no):
            valid_pts = False
            sum_loop = 0
            while valid_pts==False:
                sum_loop = sum_loop+1
                if sum_loop>100:
                    print("Stuck in the loop...........")
                    exit()
                #print("creating sample oo")
                #randomly sample a dynamic object
                samp_dyn_obj = random.sample(dynamic_obj.keys(),1)[0]

                #sample y points basedo n gaussian dist
                y_init = int(np.random.normal(dynamic_obj[samp_dyn_obj]['y_mean'], y_dist,1))

                #sample x uniformly
                x_0 = dynamic_obj[samp_dyn_obj]['x_range'][0]-x_dist
                x_1 = dynamic_obj[samp_dyn_obj]['x_range'][1]+x_dist
                x_init = random.sample(range(x_0, x_1), 1)[0]

                if y_init<0:
                    y_init = dynamic_obj[samp_dyn_obj]['y_mean'] \
                        + (dynamic_obj[samp_dyn_obj]['y_mean'] - y_init)

                if world.checkPlacementCollide(tool, [x_init,y_init]) == False:
                    valid_pts = True

                
            if tool in init_pts.keys():
                init_pts[tool].append([x_init, y_init])
            else:
                init_pts.update({tool:[[x_init, y_init]]})

    return init_pts

def gaussian_sample_policy(world, policy_params, pts_no = 4, sample=None):
    """
    The function returns a sample from a gaussian probability 
    for given mean and standard deviation
    """
    if sample==None:
        obj_list = [policy_params['w1'], policy_params['w2'], 1-policy_params['w1']-policy_params['w2']]

        tool_no = (obj_list.index(max(obj_list))+1)
        tool_no = str(tool_no)
        pt_x  = []
        pt_y = []
        for s_n in range(pts_no):
            
            x = np.random.normal(policy_params['ux'+tool_no], policy_params['sx'+tool_no])
            y = np.random.normal(policy_params['uy'+tool_no], policy_params['sy'+tool_no])
            sum_l = 0
            while world.checkPlacementCollide('obj'+tool_no, [x,y]):
                sum_l = sum_l + 1
                if sum_l > 100:
                    print("Stuck in the loop......... gaussian sample------->")
                #print("creating sample gmm")
                x = np.random.normal(policy_params['ux'+tool_no], policy_params['sx'+tool_no])
                y = np.random.normal(policy_params['uy'+tool_no], policy_params['sy'+tool_no])
        
            pt_x.append(x)
            pt_y.append(y)
            
        return {'obj'+tool_no: [[pt_x[i], pt_y[i]] for i in range(pts_no)]}
    
    else:
        tools = set(['obj1', 'obj2', 'obj3']) - set([sample])
        points = {}
        for tool in tools:
            tool_no = tool.split('obj')[1]
            
            pt_x = np.random.normal(policy_params['ux'+tool_no], policy_params['sx'+tool_no])
            pt_y = np.random.normal(policy_params['uy'+tool_no], policy_params['sy'+tool_no])
            
            while world.checkPlacementCollide(tool, [pt_x,pt_y]):
                
                pt_x = np.random.normal(policy_params['ux'+tool_no], policy_params['sx'+tool_no])
                pt_y = np.random.normal(policy_params['uy'+tool_no], policy_params['sy'+tool_no])

            points.update({tool: [[pt_x, pt_y]]})
            
        return points
            
    
def gaussian_func(mean, std, x):
    """
    The function returns the probaility of value in a 
    gaussian distribution with mean and standard deviation 
    """
    
    dist = gaussian_dist(mean, std)
    return dist.pdf(x)

def gaussian_grad_mean(mean, std, x):
    """
    The function returns the value for the gaussian gradient
    for the mean
    """
    #return prob*((std**2 - ((x-mean)**2))/std**3)
    return (x-mean)/(std**2)

def gaussian_grad_std(mean, std, x):
    """
    The function returns the value for the gaussian gradient
    for the standard deviation
    """
    #return ((((x-mean)**2 - std**2))/std**3)
    return (((x-mean)**2)/std**3)


def grad_tool_weight(mean, std, x, mean3, std3):
    """
    The function returns the gradient for the weight 
    of the tool
    """
    grad = gaussian_func(mean,std,x) - gaussian_func(mean3,std3,x)
    return grad


def find_init_dist(init_world):
    """
    The function finds the min initial distance between
    the goal and object
    Args:
    init_worls(dict): world parameters
    Return
    init_dist(float): the ditance between the values
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

    return init_goal, init_dist
    
def calc_reward(path, init_dist, goal_cord):
    """
    Normalized distance between target and reward
    args:
    world_act(dict) --> positions after the actions is performed
    init_world(dict) --> initial world position
    """
    if 'Goal' not in path.keys():
        #get the minimum distance of goal from object
        if len(path['Ball']) == 2:
            dist_list = [
                np.sqrt((cord[0]-goal_cord[0])**2 + (cord[1]-goal_cord[1])**2) \
                for cord in path['Ball'][0]
            ]
        else:
            dist_list = [
                np.sqrt((cord[0]-goal_cord[0])**2 + (cord[1]-goal_cord[1])**2) \
                for cord in path['Ball']
            ]
        
        dist_goal_min = min(dist_list)
        
        return (1-(dist_goal_min/init_dist))

    else:
        pdb.set_trace()
        

def init_policy():
    """
    The function initializes the parameters for policy of the given 
    enviornment
    """
    policy_param = {
        'w1':1, 'w2':0,
        'ux1':300, 'uy1':300, 'sx1':50**0.5, 'sy1':50**0.5,
        'ux2':300, 'uy2':300, 'sx2':50**0.5, 'sy2':50**0.5,
        'ux3':300, 'uy3':300, 'sx3':50**0.5, 'sy3':50**0.5,
        
    }
    return policy_param


def action(game_obj, init_pts):
    """
    The function takes the action in the real world 
    and returns the reward is calculated for the action
    """
    print("None")


def get_dynamic_obj(world_dict):
    """
    The function get's the information about he dynamic objects
    and their y vals
    Args:
    world_dict(Dict): Containing all the values for the world objects

    Return:
    dynamic_obj(Dict): Dictionary with dynamic values
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
                print("New structure for the dynamic object--------------")
                pdb.set_trace()

    return dynamic_obj
    
        

def simulate(game_obj, init_pts, init_dist, goal_cord, noisy=True):
    """
    Run noisy simulation based on the points to get noisy rewards
    Args:
    game_obj(dict): dictionary containing the env components
    init_pts(dict): dictionary contaiing the ponits for which the 
    simulation needs to run
    init_dist(float):initial distance bewtween goal and obj
    goal_cord(list): corrdinates for the goal
    Return:
    init_pts(dict) --> initial points for each tool
    """
    total_reward = 0
    total_cont = 0
    best_reward = -10
    for tool in init_pts.keys():
        for pts_no in init_pts[tool]:
            pts_no
            '''
            noise_dict = {
                'noise_position_static': 5., 'noise_position_moving': 5.,
                'noise_collision_direction': .2, 'noise_collision_elasticity': .2, 'noise_gravity': .1,
                'noise_object_friction': .1, 'noise_object_density': .1, 'noise_object_elasticity': .1
                         }
            '''
            noise_dict = {
                'noise_position_static': 0, 'noise_position_moving': 0,
                'noise_collision_direction': .1, 'noise_collision_elasticity': .1, 'noise_gravity': 0,
                'noise_object_friction': 0, 'noise_object_density': 0, 'noise_object_elasticity': .1
                         }
            if noisy:
                path_dict, success, time_taken, world_act = game_obj.runFullNoisyPath(
                    tool, pts_no, maxtime=20., returnDict=True, **noise_dict
                )
            else:
                path_dict, success, time_to_success = game_obj.observePlacementPath(
                    toolname=tool,position=pts_no ,maxtime=20.
                )
                if path_dict == None:
                    print("No path achieved")
                    continue
                
            reward = calc_reward(path_dict, init_dist, goal_cord)
            if reward>best_reward:
                best_reward = reward
                best = {'reward': reward, 'action': tool, 'pos': pts_no}
                
            total_reward = total_reward + reward
            total_cont = total_cont + 1

    try:
        return (total_reward/total_cont), success, best
    except ZeroDivisionError:
        return 0, success, None
    except:
        return (total_reward/total_cont), success, None
    


def get_policy_gradients(policy_params, x, y, tool):
    """
    The function calculates the gradient for a each parameter
    for the given action and tool
    """
    tool_no = tool.split('obj')[1]
    grad_w1_x = grad_tool_weight(
        policy_params['ux1'], policy_params['sx1'], x,
        policy_params['ux3'], policy_params['sx3']
    )
    grad_w1_y = grad_tool_weight(
        policy_params['uy1'], policy_params['sy1'], y,
        policy_params['uy3'], policy_params['sy3']
    )
    
    grad_w1 = (grad_w1_x + grad_w1_y)/2


    grad_w2_x = grad_tool_weight(
        policy_params['ux2'], policy_params['sx2'], x,
        policy_params['ux3'], policy_params['sx3']
    )
    grad_w2_y = grad_tool_weight(
        policy_params['uy2'], policy_params['sy2'], y,
        policy_params['uy3'], policy_params['sy3']
    )
    
    grad_w2 = (grad_w2_x + grad_w2_y)/2

    
    grad_mean_x = gaussian_grad_mean(policy_params['ux'+tool_no], policy_params['sx'+tool_no], x)
    grad_mean_y = gaussian_grad_mean(policy_params['uy'+tool_no], policy_params['sy'+tool_no], y)

    grad_std_x = gaussian_grad_std(policy_params['ux'+tool_no], policy_params['sx'+tool_no], x)
    grad_std_y = gaussian_grad_std(policy_params['uy'+tool_no], policy_params['sy'+tool_no], y)

    return {
        'w1':grad_w1, 'w2':grad_w2, 'ux'+tool_no:grad_mean_x, 'uy'+tool_no:grad_mean_y,
        'sx'+tool_no: grad_std_x, 'sy'+tool_no: grad_std_y
            }
    
def calc_prob(policy_params, tool, point):
    """
    The function calculates the policy probability for a given 
    value of points 
    """
    tool_no = tool.split('obj')[1]
    
    prob_x = gaussian_func(
        policy_params['ux'+tool_no], policy_params['sx'+tool_no], point[0])

    prob_y = gaussian_func(
        policy_params['uy'+tool_no], policy_params['sy'+tool_no], point[1])
    
    return (prob_x+prob_y)/2
    
def init_gradients():
    """
    This function creates a wrapper for the gradient values and 
    then the calculated values are addated to it
    """
    policy_grad = {
        'w1':0, 'w2':0,
        'ux1':0, 'uy1':0, 'sx1':0, 'sy1':0,
        'ux2':0, 'uy2':0, 'sx2':0, 'sy2':0,
        'ux3':0, 'uy3':0, 'sx3':0, 'sy3':0   
    }
    return policy_grad

def update_policy_params(policy_params, points, reward_all_pts):
    """
    The function updates policy parameters using policy gradients for given reward
    """
    #scale policy params and the points
    for key in policy_params:
        if (key!='w1' and key!='w2'):
            policy_params[key] = policy_params[key]/scl
    
    grads = init_gradients()
    total_prob = 0 
    for tools in points.keys():
        for point in points[tools]:
            point_0 = point[0]/scl
            point_1 = point[1]/scl
            #policy_gradient = get_policy_gradients(policy_params, point[0], point[1], tools)
            policy_gradient = get_policy_gradients(policy_params, point_0, point_1, tools)
            for key, value in policy_gradient.items():
                grads[key] = grads[key] + value
                
            #total_prob = total_prob + calc_prob(policy_params, tools, point)
            total_prob = total_prob + calc_prob(policy_params, tools, [point_0, point_1])
            
    #print("the gradients are ", grads)
    #print("total probability denominatior is ", total_prob)
    #update the policy parameters
    w1 = policy_params['w1'] + (reward_all_pts*lr*grads['w1'])
    w2 = policy_params['w2'] + (reward_all_pts*lr*grads['w2'])

    
    #if w1-w2>0.1:
    #    policy_params['w1'] = 1
    #    policy_params['w2'] = 0
    #    
    #if w2-w1>0.1:
    #    policy_params['w1'] = 0
    #    policy_params['w2'] = 1
        
    if w1>=0 and w1<=1 and (w1+w2)<=1 and (w1+w2)>=0:
        policy_params['w1'] = w1
    elif w1<0 and (w1+w2)<=1 and (w1+w2)>=0:
        policy_params['w1'] = 0
    elif w1>1 and (w1+w2)<=1 and (w1+w2)>=0:
        policy_params['w1'] = 1
        
    if w2>=0 and w2<=1 and (w1+w2)<=1 and (w1+w2)>=0:
        policy_params['w2'] = w2
    elif w2<0 and (w1+w2)<=1 and (w1+w2)>=0:
        policy_params['w2'] = 0
    elif w2>1 and (w1+w2)<=1 and (w1+w2)>=0:
        policy_params['w2'] = 1
    
    for key in grads.keys():
        if (key!='w1' and key!='w2'):
            new_param = policy_params[key] + (reward_all_pts*lr*grads[key])
            
            if (new_param<0 or new_param>(600/scl)) and 'ux' in key:
                policy_params[key] = point_0
            elif (new_param<0 or new_param>(600/scl)) and 'uy' in key:
                policy_params[key] = point_0
            elif (new_param>=0 and new_param<=(600/scl)):
                policy_params[key] = policy_params[key] + (reward_all_pts*lr*grads[key])
            #elif new_param>1:
            #    policy_params[key] = 1
            #else:
            #    policy_params[key] = policy_params[key] + (reward_all_pts*lr*grads[key])

    for key in policy_params:
        if (key!='w1' and key!='w2'):
            policy_params[key] = policy_params[key]*scl
            
    return policy_params

def SSUP_model_run(world, game, idg):
    """
    The function runs the SSUP model
    (Sample, Simulate and Update model)
    Args:
    world(dict): Contains the enviornment data values
    Return:
    model_run(Pandas df): Pandas dataframe for the model run values
    best_pos(corrdinated): best coordinate values for the given enviornment
    """
    
    #get the dynamic_objs
    dynamic_obj = get_dynamic_obj(world._worlddict['objects'])
    
    #init policy
    policy_params = init_policy()

    
    #get dist between target and object and the corrdinates for goal
    goal_cord, init_dist = find_init_dist(world._worlddict)
    

    #ssup algorithm start
    
    #Sample ninit points from prior π(s) for each tool
    init = sample(world, dynamic_obj, world._tools, 3)
    print("The initial points are ", init)

    rewards, success, best_action = simulate(world, init, init_dist, goal_cord)
    
    #Initialize policy parameters θ using policy gradient on initial points
    policy_params = update_policy_params(policy_params, init, rewards)
 

    '''
    for obj in init.keys():
        for point in init[obj]:
            #Simulate actions to get noisy rewards rˆ using internal model
            pts = {obj: [point]}
            #rewards, success, best_action = simulate(world, init, init_dist, goal_cord)
            rewards, success, best_action = simulate(world, pts, init_dist, goal_cord)
    
            #Initialize policy parameters θ using policy gradient on initial points
            #policy_params = update_policy_params(policy_params, init, rewards)
            policy_params = update_policy_params(policy_params, pts, rewards)
    '''

    
    #Give a total of 10 tries for the agent to get the best position
    #for total_tries in range(30):
    tools = ['obj1', 'obj2', 'obj3']
    ninit = 0
    trial = 0
    task_comp = False
    out_df = pd.DataFrame()
    while True:

        abs_best_action = {'reward':-1}
        if exploration():
            #sample action a from prior
            action_sample = sample(world, dynamic_obj, world._tools, 4)
            main_obj = random.sample(action_sample.keys(),1)[0]
            
            action_sample = {main_obj: action_sample[main_obj]}
            #print("Exploring!!!!!!!")

        else:
            #sample a point from the policy
            action_sample = gaussian_sample_policy(world, policy_params)
        

        avg_rewards, success, best_action = simulate(world, action_sample, init_dist, goal_cord, True)
        ninit = ninit + 1

        if best_action['reward'] > abs_best_action['reward']:
            abs_best_action = best_action
            
        if ninit==5 or avg_rewards>0.5 or success:
            ninit = 0
            trial = trial + 1
            try:
                avg_rewards, success, best_action = simulate(
                    world, {abs_best_action['action']: [abs_best_action['pos']]},
                    init_dist, goal_cord, False
                )
            except:
                pdb.set_trace()
                
            game_df = pd.DataFrame(
                data = [[
                    game, idg, trial, abs_best_action['action'],
                    abs_best_action['pos'][0], abs_best_action['pos'][1]]],
                columns = [
                    'game_name', 'game iter', 'trial_no', 'tool', 'posx', 'posy'
                ])
            out_df = pd.concat([out_df, game_df])

            #try:
            print("Demonstrating the action ", abs_best_action['action'] , abs_best_action['pos'])
            demonstrateTPPlacement(world, abs_best_action['action'] , abs_best_action['pos'])
            #except:
            #    print("Can't do the demonstration at ", abs_best_action['pos'])
            if success:
                #demonstrateTPPlacement(world, abs_best_action['action'] , abs_best_action['pos'])
                print("Successfully completed the task on trail-----> ", trial)
                task_comp = True
                return task_comp, out_df
            else:
                check_other = gaussian_sample_policy(
                    world, policy_params, sample=abs_best_action['action']
                )
                avg_rewards_other, success, best_action = simulate(
                    world, check_other,
                    init_dist, goal_cord, True
                )
                avg_rewards = (avg_rewards + avg_rewards_other)/2
                check_other.update({abs_best_action['action']: [abs_best_action['pos']]})
                policy_params = update_policy_params(policy_params, check_other, avg_rewards)
            if trial==11:
                return task_comp, out_df 
                #try:
                #    demonstrateTPPlacement(world, abs_best_action['action'] , abs_best_action['pos'])
                #except:
                #    continue
            
        else:   
            #Initialize policy parameters θ using policy gradient on initial points
            policy_params = update_policy_params(policy_params, action_sample, avg_rewards)

        if task_comp:
            break
        
        print("Trial Completed are ---> ", trial)
        print("The new policy params are", policy_params)
        print("The reward on this iteration was", avg_rewards)
