# myTeam.py
# Team Woka Woka Woka
# (Ian Galvez, Abinav Kannan, Matt Tritasavit)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [ReflexCaptureAgent(firstIndex), ReflexCaptureAgent(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.depth = 1
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        legal_actions = gameState.getLegalActions(self.index)
        next_states = [gameState.generateSuccessor(self.index, action) for action in legal_actions]
        next_index = (self.index + 1) % gameState.getNumAgents()
        action_values = [self.get_value(next_state, self.depth, next_index) for next_state in next_states]

        best_value = max(action_values)
        index_of_best_value = action_values.index(best_value)
        return legal_actions[index_of_best_value]

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    def evaluate(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)
        return features * weights

    def getFeatures(self, gameState):
        # OFFENSIVE FEATURES
        # ourTeamFoodCount - length of the food list
        # distanceToFood - which is the maze distance to the closest food
        # numGhosts - number of ghosts
        # ghostDistance - distance to the closest ghost
        # deadEnd - if in the next state you get into a dead end, don't go inside
        # getback - make him return after you've eaten enough pellets
        features = util.Counter()
        foodList = self.getFood(gameState).asList()
        features['enemyTeamFoodCount'] = len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = gameState.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numGhosts'] = len(ghosts)
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            features['ghostDistance'] = min(dists)

        escape = [gameState.getLegalActions()]
        if len(ghosts) > 0:
            ghost_dist = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            min_ghost_dist = min(ghost_dist)
            if len(escape) == 1 and min_ghost_dist < 4:
                features['deadEnd'] = 1
            else:
                features['deadEnd'] = 0
        '''
        if gameState.getScore() > 2:
            features['getBack'] = 1
        else:
            features['getBack'] = 0
        '''
        
        # DEFENSIVE FEATURES
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        foodDefenseList = self.getFoodYouAreDefending(gameState).asList()
        features['ourTeamFoodCount'] = len(foodDefenseList)  # self.getScore(successor)

        return features

    def getWeights(self, gameState):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """

        if gameState.getScore() > 0:  # defense
            return {'numInvaders': -1000, 'invaderDistance': -10, 'ourTeamFoodCount': 100}
        else:                         # offense
            return {'enemyTeamFoodCount': -100, 'distanceToFood': -1, 'numInvaders': -100,
                    'invaderDistance': -10, 'deadEnd': -10000, 'ghostDistance': -10,
                    'numGhosts': -100}

    def get_value(self, state, depth, index):
        if index == min(self.getTeam(state)):
            depth -= 1

        if depth == 0 or state.isOver():
            return self.evaluate(state)
        if index in self.getTeam(state):
            return self.max_value(state, depth, index)
        else:
            return self.min_value(state, depth, index)

    def max_value(self, state, depth, index):
        next_states = [state.generateSuccessor(index, action) for action in state.getLegalActions(index)]
        next_index = (index + 1) % state.getNumAgents()
        action_values = [self.get_value(next_state, depth, next_index) for next_state in next_states]
        return max(action_values)

    def min_value(self, state, depth, index):
        next_states = [state.generateSuccessor(index,action) for action in state.getLegalActions(index)]
        next_index = (index + 1) % state.getNumAgents()
        action_values = [self.get_value(next_state, depth, next_index) for next_state in next_states]
        return min(action_values)
