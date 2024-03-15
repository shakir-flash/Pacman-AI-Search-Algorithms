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


import random
import util

from game import Agent


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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Initialize an evaluation score parameter
        evalScore = successorGameState.getScore()

        #Calculate manhattan distances to food modules
        foodDistances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]

        #Calculate distance to the closest food using min function
        minFoodDistance = min(foodDistances) if foodDistances else 0

        #Calculate distances to ghost agents
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        #Check if Pacman is in danger from ghost using ghostDistances
        if min(ghostDistances) < 2 and min(newScaredTimes) == 0:
            evalScore -= 100  # Penalize if a ghost is too close and Pacman is not scared

        #Check if Pacman can eat a ghost
        if min(ghostDistances) < 2 and min(newScaredTimes) > 0:
            evalScore += 200  # Reward if a ghost is too close and Pacman is scared

        #Add score based on the reciprocal of the distance to the closest food
        evalScore += 1.0 / (minFoodDistance + 1) #Added 1 since there are instances of division by 0.

        return evalScore


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def minimize(gameState, depth, agentIndex):
        #if the game is won or lost, return the evaluation function value
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            #initialize the minimum value to a high positive value
            min_value = float('inf')
            
            #if it's the last agent's turn, calculate the minimum value based on the maximum level
            if agentIndex == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    min_value = min(min_value, maximize(successor, depth, 0))
            # if it's not the last agent's turn, recursively call minimize for the next agent
            else:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    min_value = min(min_value, minimize(successor, depth, agentIndex + 1))
            
            return min_value

        def maximize(gameState, depth, agentIndex):
            #increment values for depth iteration
            current_depth = depth + 1
            
            #if the game is won, lost, or depth limit is reached, return the evaluation function value
            if gameState.isLose() or gameState.isWin() or current_depth == self.depth:
                return self.evaluationFunction(gameState)
            
            #initialize the maximum value to high negative number
            max_value = float('-inf')   
            
            # loop over legal actions for the agent
            for action in gameState.getLegalActions(agentIndex):
                successor_state = gameState.generateSuccessor(agentIndex, action)
                max_value = max(max_value, minimize(successor_state, current_depth, 1))
            
            return max_value

        # selection logic
        current_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            #generate the next state after taking the action
            next_state = gameState.generateSuccessor(0, action)
            # calculate the score using the minimize function for the next level
            score = minimize(next_state, 0, 1)
            # choose the action with the maximum score
            if score > current_score:
                best_action = action
                current_score = score

        # return the best action found
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def minimize(gameState, depth, agentIndex, alpha, beta):
            #if the game is won or lost, return the evaluation function value
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            #initialize the minimum value to a high positive value
            min_value = float('inf')
            
            #if it's the last agent's turn, calculate the minimum value based on the maximum level
            if agentIndex == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    min_value = min(min_value, maximize(successor, depth, 0, alpha, beta))
                    if min_value < alpha:
                        return min_value
                    beta = min(beta, min_value)
            # if it's not the last agent's turn, recursively call minimize for the next agent
            else:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    min_value = min(min_value, minimize(successor, depth, agentIndex + 1, alpha, beta))
                    if min_value < alpha:
                        return min_value
                    beta = min(beta, min_value)
            
            return min_value

        # function to find the maximum value for the current state and agent
        def maximize(gameState, depth, agentIndex, alpha, beta):
            #increment values for depth iteration
            current_depth = depth + 1
            
            #if the game is won, lost, or depth limit is reached, return the evaluation function value
            if gameState.isLose() or gameState.isWin() or current_depth == self.depth:
                return self.evaluationFunction(gameState)
            
            #initialize the maximum value to high negative number
            max_value = float('-inf')
            
            # loop over legal actions for the agent
            for action in gameState.getLegalActions(agentIndex):
                successor_state = gameState.generateSuccessor(agentIndex, action)
                max_value = max(max_value, minimize(successor_state, current_depth, 1, alpha, beta))
                if max_value > beta:
                    return max_value
                alpha = max(alpha, max_value)
            
            return max_value

        # Selection logic
        current_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            #generate the next state after taking the action
            next_state = gameState.generateSuccessor(0, action)
            # calculate the score using the minimize function for the next level
            score = minimize(next_state, 0, 1, alpha, beta)
            # choose the action with the maximum score
            if score > current_score:
                best_action = action
                current_score = score
            alpha = max(alpha, score)

        return best_action



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
        def maximize(gameState, depth, agentIndex):
            #if the game is won, lost, or depth limit is reached, return the evaluation function value
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            #initialize the maximum value to high negative number
            max_score = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                #generate the successor state after taking the action
                successor_state = gameState.generateSuccessor(agentIndex, action)
                #update  maximum score based on the expected value of the successor state
                max_score = max(max_score, expect_value(successor_state, depth, 1))
            return max_score

        def expect_value(gameState, depth, agentIndex):
            #if the game is won or lost, return the evaluation function value
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            #get legal actions for the current agent
            legal_actions = gameState.getLegalActions(agentIndex)
            num_actions = len(legal_actions)
            total_score = 0
            
            #iterate over legal actions for the current agent
            for action in legal_actions:
                #generate the successor state after taking the action
                successor_state = gameState.generateSuccessor(agentIndex, action)
                
                #recursively calculate the score from successor states
                if agentIndex == gameState.getNumAgents() - 1:
                    total_score += maximize(successor_state, depth + 1, 0)
                else:
                    total_score += expect_value(successor_state, depth, agentIndex + 1)
            
            return total_score / num_actions

        actions = gameState.getLegalActions(0)
        best_action = None
        best_score = float('-inf')
        
        #iterate over legal actions for Pacman
        for action in actions:
            #calculate the expected value for each action
            action_score = expect_value(gameState.generateSuccessor(0, action), 0, 1)
            #update the best action and score based on the action with the highest expected value
            if action_score > best_score:
                best_action = action
                best_score = action_score
        return best_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    #get relevant information from the current game state
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    #check if Pacman has eaten a capsule
    hasEatenCapsule = currentGameState.getNumAgents() > 1 and currentGameState.data.agentStates[0].scaredTimer > 0

    #calculate the closest food distance if there is remaining food
    remainingFood = foodGrid.asList()
    if remainingFood:
        closestFoodDistance = min(util.manhattanDistance(pacmanPosition, food) for food in remainingFood)
    else:
        closestFoodDistance = 0

    #calculate the closest ghost distance
    closestGhostDistance = min(util.manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates)

    #evaluate the remaining food and capsules
    remainingFoodCount = len(remainingFood)
    remainingCapsules = len(capsules)

    #evaluate the game score
    score = currentScore

    #penalty for the number of remaining food pellets
    foodPenalty = -100 * remainingFoodCount

    #penalty for the number of remaining capsules
    capsulePenalty = -1000 * remainingCapsules

    foodWeight = 10  #weight for food pellets
    ghostWeight = -20  #weight for avoiding ghosts
    chaseGhostWeight = 500  #weight for chasing ghosts after eating capsules

    #combine the factors into an evaluation score
    if hasEatenCapsule:
        #pacman has eaten a capsule, prioritize chasing ghosts
        evaluation = score + chaseGhostWeight / (closestGhostDistance + 1) + foodPenalty + capsulePenalty
    else:
        #pacman has not eaten a capsule, prioritize eating food pellets and avoiding ghosts
        evaluation = score + foodWeight / (closestFoodDistance + 1) + ghostWeight / (closestGhostDistance + 1) + foodPenalty + capsulePenalty
    return evaluation

# Abbreviation
better = betterEvaluationFunction
