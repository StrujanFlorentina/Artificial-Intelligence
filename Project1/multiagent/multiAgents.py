# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class RandomAgent(Agent):
    def getAction(selfself, gameState):
        legalMoves = gameState.getLegalActions()
        chosenAction = random.choice(legalMoves)
        return chosenAction


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # score food location and ghost location in order to select a better action

        distToFood = []
        distToGhost = []

        # Find the distance of all the foods to the pacman
        for food in newFood:
            distance = manhattanDistance(food, newPos)
            distToFood.append(distance)

        # Find the distance of all the ghost to the pacman
        for ghost in newGhostPositions:
            distanceG = manhattanDistance(ghost, newPos)
            distToGhost.append(distanceG)

        minGhost = min(distToGhost)

        for ghostDistance in distToGhost:
            if ghostDistance < 2:
                return -999999
            else:
                if len(distToFood) != 0:
                    minFoodDist = min(distToFood)
                else:
                    return 999999

        return successorGameState.getScore() + 1.0 / (minFoodDist + minGhost)

        "*** YOUR CODE HERE ***"


"""
        print("current position: ", currentGameState.getPacmanPosition())
        print("succesor: ", newPos," with action: ", action," and score: ", successorGameState.getScore())
        print("food left: ", newFood.asList())
        #daca mananca un power up, poate manca dupa temporar o fantoma
        print("Scared time: ",newScaredTimes)
        ghostPositions = currentGameState.getGhostPositions()
        distToGhost = [manhattanDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
        distToFood=[manhattanDistance(newPos,foodPos) for foodPos in newFood.asList()]


        d1=min(distToFood)
        d2= min(distToGhost)


        print("current position: ", currentGameState.getPacmanPosition()," with action: ", action," results in: ", newPos)
        print("Ghost position: ",ghostPositions)
        print("Distance from Pacman to Ghosts: ", distToGhost)
        print("----------------------------------------------------------------------------")

        return successorGameState.getScore()"""


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        result = self.minimax(gameState, 0, 0)
        # Return the action from result
        return result[1]

    def minimax(self, gameState, agentIndex, depth):
        bestAction = ""
        legalActions = gameState.getLegalActions(agentIndex)

        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), bestAction

        # pacman
        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)

        # ghost
        else:
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):

        legalAction = gameState.getLegalActions(agentIndex)
        value = -999999
        bestAction = ""

        for action in legalAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            # Update the successor agent's index and depth if it's pacman
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            current_value = self.minimax(successor, successorIndex, successorDepth)[0]

            if current_value > value:
                value = current_value
                bestAction = action

        return value, bestAction

    def min_value(self, gameState, agentIndex, depth):

        legalMoves = gameState.getLegalActions(agentIndex)
        value = 999999
        bestAction = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            # Update the successor agent's index and depth if it's pacman
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            current_value = self.minimax(successor, successorIndex, successorDepth)[0]

            if current_value < value:
                value = current_value
                bestAction = action

        return value, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -999999
        beta = 999999
        result = self.abpruning(gameState, 0, 0, alpha, beta)

        return result[1]

    def abpruning(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ""
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), bestAction

        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)

    def max_value(self, gameState, agentIndex, depth, alpha, beta):

        legalActions = gameState.getLegalActions(agentIndex)
        value = -999999
        bestAction = ""

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            current_value, current_action \
                = self.abpruning(successor, successorIndex, successorDepth, alpha, beta)

            if current_value > value:
                value = current_value
                bestAction = action

            alpha = max(alpha, value)

            if value > beta:
                return value, bestAction

        return value, bestAction

    def min_value(self, gameState, agentIndex, depth, alpha, beta):

        legalActions = gameState.getLegalActions(agentIndex)
        value = 999999
        bestAction = ""

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            current_value, current_action \
                = self.abpruning(successor, successorIndex, successorDepth, alpha, beta)

            if current_value < value:
                value = current_value
                bestAction = action

            beta = min(beta, value)

            if value < alpha:
                return value, bestAction

        return value, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
