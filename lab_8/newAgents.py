from pacman import Directions
from game import Agent
import game
import util

import random
import numpy as np
import math

class NewAgent1(Agent):
    def setCode(self,codep):
        self.code = codep
    
    def getAction(self,state):
        px,py = state.getPacmanPosition()
        
        # g1x,g1y= state.getGhostPosition(1)
        # ghost1Angle = np.arctan2(g1y-py,g1x-px)
        # if ghost1Angle<0.0:
        #     ghost1Angle += 2.0*math.pi
        # ghost1Dist = math.floor(np.sqrt( (g1x-px)**2 + (g1y-py)**2 ))
        # ghost1Pos = ""
        # if math.pi/4.0 < ghost1Angle <= 3.0*math.pi/3.0:
        #     ghost1Pos = "up"
        # if 3.0*math.pi/4.0 < ghost1Angle <= 5.0*math.pi/3.0:
        #     ghost1Pos = "left"
        # if 5.0*math.pi/4.0 < ghost1Angle <= 7.0*math.pi/3.0:
        #     ghost1Pos = "down"
        # if 7.0*math.pi/4.0 < ghost1Angle <= 2.0*math.pi:
        #     ghost1Pos = "right"
        # if 0.0 <= ghost1Angle <= math.pi/4.0:
        #     ghost1Pos = "right"
        #print(ghost1Angle," ",ghost1Pos)
        #print(ghost1Dist)
        #print(state.getCapsules())
        #print(state.getFood())

        ch = self.code[px][py]
        legal = state.getLegalPacmanActions()
        if ch not in legal:
            ch = random.choice(legal)
        return ch
