from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement
import json
import pygame as pg

# Make the basic world
pgw = PGWorld(dimensions=(600,600), gravity=200)
# Name, [left, bottom, right, top], color, density (0 is static)
pgw.addBox('Table', [0,0,300,200],(0,0,0),0)
# Name, points (counter-clockwise), width, color, density
pgw.addContainer('Goal', [[330,100],[330,5],[375,5],[375,100]], 10, (0,255,0), (0,0,0), 0)
# Name, position of center, radius, color, (density is 1 by default)
pgw.addBall('Ball',[100,215],15,(0,0,255))

# Sets up the condition that "Ball" must go into "Goal" and stay there for 2 seconds
pgw.attachSpecificInGoal("Goal","Ball",2.)
pgw_dict = pgw.toDict()

tools = {
    "obj1" : [[[-30,-15],[-30,15],[30,15],[0,-15]]],
    "obj2" : [[[-20,0],[0,20],[20,0],[0,-20]]],
    "obj3" : [[[-40,-5],[-40,5],[40,5],[40,-5]]]
    }

tp = ToolPicker(
    {'world': pgw_dict,
     'tools': tools}
)

path_dict, success, time_to_success = tp.observePlacementPath(toolname="obj1",position=(90,400),maxtime=20.)
print("Action was successful? ", success)