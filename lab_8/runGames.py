import matplotlib.pyplot as plt
from pacman import *
import ghostAgents
import layout
import textDisplay
import graphicsDisplay
import copy
import numpy as np
from pprint import pprint
import sys

## set up the parameters to newGame
numtraining = 0
timeout = 30
beQuiet = False
layout=layout.getLayout("mediumClassic")
pacmanType = loadAgent("NewAgent1", True)
numGhosts = 1
ghosts = [ghostAgents.RandomGhost(i+1) for i in range(numGhosts)]
catchExceptions=True

def run(code,noOfRuns):
    rules = ClassicGameRules(timeout)
    games = []
    if beQuiet:
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        timeInterval = 0.001
        textDisplay.SLEEP_TIME = timeInterval
        gameDisplay = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
        rules.quiet = False
    for gg in range(noOfRuns):
        thePacman = pacmanType()
        thePacman.setCode(code)
        game = rules.newGame( layout, thePacman, ghosts, gameDisplay, \
                          beQuiet, catchExceptions )
        game.run()
        games.append(game)
    scores = [game.state.getScore() for game in games]
    return sum(scores) / float(len(scores))

####### genetic algorithm

options = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
    
def mutate(parentp,numberOfMutations=10):
    parent = copy.deepcopy(parentp)
    for _ in range(numberOfMutations):
        xx = random.randrange(19)
        yy = random.randrange(10)
        parent[xx][yy] = random.choice(options)
    return parent

def runGA(popSiz=20,timescale=20,numberOfRuns=2,tournamentSize=4):

    ## create random initial population
    population = []
    for _ in range(popSiz):
        program = np.empty((19,10),dtype=object)
        for xx in range(19):
            for yy in range(10):
                program[xx][yy] = random.choice(options)
        population.append(program)
            
    print("Beginning Evolution")
    averages = []
    bests = []
    for _ in range(timescale):
        ## evaluate population
        fitness = []
        for pp in population:
            print(".",end="",flush=True)
            fitness.append(run(pp,numberOfRuns))
        print("\n******")
        print(fitness)
        averages.append(1000+sum(fitness)/popSiz)
        print("av ",1000+sum(fitness)/popSiz)
        bests.append(1000+max(fitness))
        print("max ",1000+max(fitness))

        popFitPairs = list(zip(population,fitness))
        newPopulation = []
        for _ in range(popSiz-1):
                # select a parent from a "tournament"
                tournament = random.sample(popFitPairs,tournamentSize)
                parent = max(tournament,key=lambda x:x[1])[0]
                # mutate the parent
                child = mutate(parent)
                newPopulation.append(child)
                ## ADD CODE TO CROSSOVER PARENTS
        ## ADD CODE TO KEEP BEST POPULATION MEMBER
        population = copy.deepcopy(newPopulation)
    print(averages)
    print(bests)

    ## ADD CODE TO PLOT averages AND bests

def runTest():
    program = np.empty((19,10),dtype=object)
    for xx in range(19):
        for yy in range(10):
            program[xx][yy] = Directions.EAST
    
    run(program,1)

runTest()    
#runGA()
        

