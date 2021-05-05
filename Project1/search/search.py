# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random


class CustomNode:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def getState(self):
        return self.state

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def getCost(self):
        return self.cost


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def randomSearch(problem):
    solutions = []
    currentState = problem.getStartState()
    while not problem.isGoalState(currentState):
        succesors = problem.getSuccessors(currentState)
        numberOfSuccessors = len(succesors)
        randomIndex = random.radint(0, numberOfSuccessors - 1)
        randomSuccessor = succesors[randomIndex]
        action = randomSuccessor[0]
        solutions.append(action)
    print("the solution is: ", solutions)
    return solutions


def depthFirstSearch(problem):
    """ from game import Directions
        w = Directions.WEST
        s=Directions.SOUTH
        return [w,w,w,w,w,s]"""

    """
        Search the deepest nodes in the search tree first.

        Your search algorithm needs to return a list of actions that reaches the
        goal. Make sure to implement a graph search algorithm.

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:



        "*** YOUR CODE HERE ***"
        print("Start:", problem.getStartState())
        print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        for (state, action, cost) in problem.getSuccessors(problem.getStartState()):
            print("Start's successors: State:= ", state, " action: ", action, " cost", cost)
       #
    """

    """
        # Get the succesors of the initial state and print the state, the action and the cost for each successor. Run again with smallMaze.

        initialState=problem.getStartState()
        tupleList=[]

        i=0
        while i<2:
            (state, action, _) = problem.getSuccessors(initialState)[i]
            (nextState, nextAction, _) = problem.getSuccessors(state)[i]
            i += 1
            tupleList.append((nextState, nextAction, _))
        print("Succesors of initial state: ", tupleList)
    """

    """
    node1=CustomNode("firstNode",3)
    node2=CustomNode("secondNode",5)
    myStack=util.Stack()
    myStack.push(node1)
    myStack.push(node2)

    extractedNode=myStack.pop()
    print("Extracted node is: ",extractedNode.getName())
"""
    startingNode = problem.getStartState()

    if problem.isGoalState(startingNode):
        return []
    myStack = util.Stack()
    visitedNodes = []

    node = CustomNode(startingNode, None, [], 0)
    myStack.push(node)

    while not myStack.isEmpty():
        currentNode = myStack.pop()

        if currentNode.getState() not in visitedNodes:
            visitedNodes.append(currentNode.getState())
            if problem.isGoalState(currentNode.getState()):
                return currentNode.getAction()

            for node, action, cost in problem.getSuccessors(currentNode.getState()):
                newAction = currentNode.getAction() + [action]
                newCost = currentNode.getCost() + cost
                newNode = CustomNode(node, currentNode.getState(), newAction, newCost)
                myStack.push(newNode)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    startingNode = problem.getStartState()

    if problem.isGoalState(startingNode):
        return []
    myQueue = util.Queue()
    visitedNodes = []

    node = CustomNode(startingNode, None, [], 0)
    myQueue.push(node)

    while not myQueue.isEmpty():
        currentNode = myQueue.pop()

        if currentNode.getState() not in visitedNodes:
            visitedNodes.append(currentNode.getState())
            if problem.isGoalState(currentNode.getState()):
                return currentNode.getAction()

            for node, action, cost in problem.getSuccessors(currentNode.getState()):
                newAction = currentNode.getAction() + [action]
                newCost = currentNode.getCost() + cost
                newNode = CustomNode(node, currentNode.getState(), newAction, newCost)
                myQueue.push(newNode)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    startingNode = problem.getStartState()
    # priority queue ordered by path cost
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []

    if problem.isGoalState(startingNode):
        return []

    node = CustomNode(startingNode, None, [], 0)
    myPriorityQueue.push(node, 0)

    while not myPriorityQueue.isEmpty():
        currentNode = myPriorityQueue.pop()  # lowest cost
        previousState = currentNode.getState()
        previousAction = currentNode.getAction()
        previousCost = currentNode.getCost()

        if problem.isGoalState(previousState):
            return previousAction
        if previousState not in visitedNodes:
            visitedNodes.append(previousState)

            successors = problem.getSuccessors(previousState)

            for state, action, cost in successors:
                newAction = previousAction + [action]
                newCost = previousCost + cost
                newNode = CustomNode(state, previousState, newAction, newCost)
                myPriorityQueue.push(newNode, newCost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startingNode = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []

    if problem.isGoalState(startingNode):
        return []

    node = CustomNode(startingNode, None, [], 0)
    myPriorityQueue.push(node, 0)

    while not myPriorityQueue.isEmpty():
        currentNode = myPriorityQueue.pop()  # lowest cost
        previousState = currentNode.getState()
        previousAction = currentNode.getAction()
        previousCost = currentNode.getCost()

        if problem.isGoalState(previousState):
            return previousAction
        if previousState not in visitedNodes:
            visitedNodes.append(previousState)

            successors = problem.getSuccessors(previousState)

            for state, action, cost in successors:
                newAction = previousAction + [action]
                newCost = previousCost + cost
                heuristicCost = newCost + heuristic(state, problem)
                newNode = CustomNode(state, previousState, newAction, newCost)
                myPriorityQueue.push(newNode, heuristicCost)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
rs = randomSearch
# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random


class CustomNode:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def getState(self):
        return self.state

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def getCost(self):
        return self.cost


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def randomSearch(problem):
    solutions = []
    currentState = problem.getStartState()
    while not problem.isGoalState(currentState):
        succesors = problem.getSuccessors(currentState)
        numberOfSuccessors = len(succesors)
        randomIndex = random.radint(0, numberOfSuccessors - 1)
        randomSuccessor = succesors[randomIndex]
        action = randomSuccessor[0]
        solutions.append(action)
    print("the solution is: ", solutions)
    return solutions


def depthFirstSearch(problem):
    """ from game import Directions
        w = Directions.WEST
        s=Directions.SOUTH
        return [w,w,w,w,w,s]"""

    """
        Search the deepest nodes in the search tree first.

        Your search algorithm needs to return a list of actions that reaches the
        goal. Make sure to implement a graph search algorithm.

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:



        "*** YOUR CODE HERE ***"
        print("Start:", problem.getStartState())
        print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        for (state, action, cost) in problem.getSuccessors(problem.getStartState()):
            print("Start's successors: State:= ", state, " action: ", action, " cost", cost)
       #
    """

    """
        # Get the succesors of the initial state and print the state, the action and the cost for each successor. Run again with smallMaze.

        initialState=problem.getStartState()
        tupleList=[]

        i=0
        while i<2:
            (state, action, _) = problem.getSuccessors(initialState)[i]
            (nextState, nextAction, _) = problem.getSuccessors(state)[i]
            i += 1
            tupleList.append((nextState, nextAction, _))
        print("Succesors of initial state: ", tupleList)
    """

    """
    node1=CustomNode("firstNode",3)
    node2=CustomNode("secondNode",5)
    myStack=util.Stack()
    myStack.push(node1)
    myStack.push(node2)

    extractedNode=myStack.pop()
    print("Extracted node is: ",extractedNode.getName())
"""
    startingNode = problem.getStartState()

    if problem.isGoalState(startingNode):
        return []
    myStack = util.Stack()
    visitedNodes = []

    node = CustomNode(startingNode, None, [], 0)
    myStack.push(node)

    while not myStack.isEmpty():
        currentNode = myStack.pop()

        if currentNode.getState() not in visitedNodes:
            visitedNodes.append(currentNode.getState())
            if problem.isGoalState(currentNode.getState()):
                return currentNode.getAction()

            for node, action, cost in problem.getSuccessors(currentNode.getState()):
                newAction = currentNode.getAction() + [action]
                newCost = currentNode.getCost() + cost
                newNode = CustomNode(node, currentNode.getState(), newAction, newCost)
                myStack.push(newNode)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    startingNode = problem.getStartState()

    if problem.isGoalState(startingNode):
        return []
    myQueue = util.Queue()
    visitedNodes = []

    node = CustomNode(startingNode, None, [], 0)
    myQueue.push(node)

    while not myQueue.isEmpty():
        currentNode = myQueue.pop()

        if currentNode.getState() not in visitedNodes:
            visitedNodes.append(currentNode.getState())
            if problem.isGoalState(currentNode.getState()):
                return currentNode.getAction()

            for node, action, cost in problem.getSuccessors(currentNode.getState()):
                newAction = currentNode.getAction() + [action]
                newCost = currentNode.getCost() + cost
                newNode = CustomNode(node, currentNode.getState(), newAction, newCost)
                myQueue.push(newNode)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    startingNode = problem.getStartState()
    # priority queue ordered by path cost
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []

    if problem.isGoalState(startingNode):
        return []

    node = CustomNode(startingNode, None, [], 0)
    myPriorityQueue.push(node, 0)

    while not myPriorityQueue.isEmpty():
        currentNode = myPriorityQueue.pop()  # lowest cost
        previousState = currentNode.getState()
        previousAction = currentNode.getAction()
        previousCost = currentNode.getCost()

        if problem.isGoalState(previousState):
            return previousAction
        if previousState not in visitedNodes:
            visitedNodes.append(previousState)

            successors = problem.getSuccessors(previousState)

            for state, action, cost in successors:
                newAction = previousAction + [action]
                newCost = previousCost + cost
                newNode = CustomNode(state, previousState, newAction, newCost)
                myPriorityQueue.push(newNode, newCost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startingNode = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []

    if problem.isGoalState(startingNode):
        return []

    node = CustomNode(startingNode, None, [], 0)
    myPriorityQueue.push(node, 0)

    while not myPriorityQueue.isEmpty():
        currentNode = myPriorityQueue.pop()  # lowest cost
        previousState = currentNode.getState()
        previousAction = currentNode.getAction()
        previousCost = currentNode.getCost()

        if problem.isGoalState(previousState):
            return previousAction
        if previousState not in visitedNodes:
            visitedNodes.append(previousState)

            successors = problem.getSuccessors(previousState)

            for state, action, cost in successors:
                newAction = previousAction + [action]
                newCost = previousCost + cost
                heuristicCost = newCost + heuristic(state, problem)
                newNode = CustomNode(state, previousState, newAction, newCost)
                myPriorityQueue.push(newNode, heuristicCost)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
rs = randomSearch
