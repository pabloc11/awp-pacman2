# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint
import regularMutation
from game import Actions

#############
# FACTORIES #
#############

class GreatAgents(AgentFactory):
  "Returns three offensive great agents"
  
  def __init__(self, isRed, **args):
    AgentFactory.__init__(self, isRed)
    self.agents = ["goal", "defense", "defense"]
    self.teamData = TeamData() 
  def getAgent(self, index):
      return self.choose(self.agents.pop(0), index)

  def choose(self, agentStr, index):
    if agentStr == 'offense':
      return OffensiveGreatAgent(index, self.teamData)
    elif agentStr == 'defense':
      return DefensiveGreatAgent(index, self.teamData)
    elif agentStr == 'experimental':
      return ExperimentalAgent(index, self.teamData)
    elif agentStr == 'goal':
      return GoalBasedAgent(index, self.teamData)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class TeamData:
  def __init__(self):
    self.initialized = False
    self.legalPositions = []
    self.opponentBeliefs = []
    self.opponentPositions = []
    self.deadEnds = util.Counter()
    self.goal = None

##########
# Agents #
##########

class GreatAgent(CaptureAgent):
  def __init__(self, index, teamData):
    CaptureAgent.__init__(self, index)
    self.firstTurnComplete = False
    self.startingFood = 0
    self.theirStartingFood = 0
    self.teamData = teamData
  
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    
    If you want initialization any items a single time for a team
    (i.e. the first agent that gets created does the work) you'll
    need to create your own team class and only let it initialize
    once.
    
    A distanceCalculator instance caches the maze distances 
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    """
    self.red = gameState.isOnRedTeam(self.index)
    # Even though there are up to 6 agents creating a distancer, the distances
    # will only actually be computed once, before the start of the game 
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    
    # comment this out to forgo maze distance computation and use manhattan distances
    self.distancer.getMazeDistances()
    
    # or uncomment this to forget maze distances and just use manhattan distances for this agent
    #self.distancer.useManhattanDistances()
    
    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display
    
    self.walls = gameState.getWalls()
    
    if self.teamData.initialized == False:
      self.initializeTeamData(gameState)
      self.teamData.initialized = True

  
  def initializeTeamData(self, gameState):
    # Find all possible legal positions
    width = self.getLayoutWidth(gameState)
    height = self.getLayoutHeight(gameState)
    beliefs = util.Counter()
    for i in range(0, width):
      for j in range(0, height):
        if gameState.hasWall(i,j) == False:
          self.teamData.legalPositions.append((i,j))
          beliefs[(i,j)] = 1
    beliefs.normalize()
                
    # set up beliefs for each opponent agent
    self.teamData.opponentPositions = self.getOpponentPositions(gameState)
    
    for opponentIndex in self.getOpponents(gameState):
      self.teamData.opponentBeliefs.append(beliefs.copy())
      
    # Find all dead ends in the maze
    numLegalNeighbors = util.Counter()
    possibleDeltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    frontier = []
    
    for p in self.teamData.legalPositions:
      numLegalNeighbors[p] = 0
      for dx, dy in possibleDeltas:
        if not gameState.hasWall(p[0] + dx, p[1] + dy):
          numLegalNeighbors[p] += 1
      if numLegalNeighbors[p] == 1:
        frontier.append(p)
    
    while len(frontier) > 0:
      cur = frontier.pop(0)
      numNonDeadEndNeighbors = 0
      for dx, dy in possibleDeltas:
        neighbor = (cur[0] + dx, cur[1] + dy)
        if not gameState.hasWall(neighbor[0], neighbor[1]) and self.teamData.deadEnds[neighbor] == 0:
          numNonDeadEndNeighbors += 1
      if numNonDeadEndNeighbors == 1:
        self.increment(cur, util.Counter())
        self.teamData.deadEnds[cur] = 1
        for dx, dy in possibleDeltas:
          neighbor = (cur[0] + dx, cur[1] + dy)
          if not gameState.hasWall(neighbor[0], neighbor[1]) and self.teamData.deadEnds[neighbor] == 0:
            frontier.append(neighbor)    

  def increment(self, position, alreadyVisited):
    if self.teamData.deadEnds[position] > 0:
      self.teamData.deadEnds[position] += 1
      alreadyVisited[position] = 1
    
    possibleDeltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in possibleDeltas:
      neighbor = (position[0] + dx, position[1] + dy)
      if self.teamData.deadEnds[neighbor] > 0 and alreadyVisited[neighbor] == 0:
        self.increment(neighbor, alreadyVisited)
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    if not self.firstTurnComplete:
      self.firstTurnComplete = True
      self.startingFood = len(self.getFoodYouAreDefending(gameState).asList())
      self.theirStartingFood = len(self.getFood(gameState).asList())
    
    self.updateBeliefs(gameState)
    
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def updateBeliefs(self, gameState):
    '''
    Update beliefs on opponents' positions
    '''
    
    myPosition = self.getPosition(gameState)
    opponentPositions = self.getOpponentPositions(gameState)
    opponents = self.getOpponents(gameState)
    team = self.getTeam(gameState)
    teamPositions = self.getTeamPositions(gameState)
    
    for i, position in enumerate(opponentPositions):
      beliefs = self.teamData.opponentBeliefs[i]     
        
      # beliefs are re-initialized if pacman is captured
      if beliefs[beliefs.argMax()] == 0:
        for pos in self.teamData.legalPositions:
          beliefs[pos] = 1
        beliefs.normalize()
          
      if position == None:
        oldBeliefs = beliefs.copy()
        noisyDistance = gameState.getAgentDistance(opponents[i])
        for pos in self.teamData.legalPositions:
          trueDistance = util.manhattanDistance(myPosition, pos)
          distanceProb = gameState.getDistanceProb(trueDistance, noisyDistance)
          if distanceProb == 0:
            beliefs[pos] = 0
          else:
            sum = 0
            # only account for an enemy moving when it actually moves, not every time you make an observation
            if (self.index - 1 == opponents[i]) or ((self.index == 0) and (i + 1 == len(opponents))):
              previousPos = Actions.getLegalNeighbors(pos, self.walls)
            else:
              previousPos = [pos]
            for prePos in previousPos:
              sum += oldBeliefs[prePos]
            beliefs[pos] += sum / len(previousPos) 
            beliefs[pos] *= distanceProb
      else:
        beliefs.clear()
        for pos in self.teamData.legalPositions:
          beliefs[pos] = 0
        beliefs[position] = 1
      beliefs.normalize()
      self.teamData.opponentPositions[i] = beliefs.argMax()
    
    # print out opponents' positions (max prob)
    #if self.index == 0 or self.index == 1:
    self.displayDistributionsOverPositions(self.teamData.opponentBeliefs)
    #  for index, pos in enumerate(self.opponentPositions):
    #    print opponents[index], pos, self.opponentBeliefs[index]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  
  """
  Features (not the best features) which have learned weight values stored.
  """
  def getMutationFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    position = self.getPosition(gameState)

    distances = 0.0
    for tpos in self.getTeamPositions(successor):
      distances = distances + abs(tpos[0] - position[0])
    features['xRelativeToFriends'] = distances
    
    enemyX = 0.0
    for epos in self.teamData.opponentPositions:
      if epos is not None:
        enemyX = enemyX + epos[0]
    features['avgEnemyX'] = distances
    
    foodLeft = len(self.getFoodYouAreDefending(successor).asList())
    features['percentOurFoodLeft'] = foodLeft / self.startingFood
    
    foodLeft = len(self.getFood(successor).asList())
    features['percentTheirFoodLeft'] = foodLeft / self.theirStartingFood
    
    features['IAmAScaredGhost'] = 1.0 if self.isPacman(successor) and self.getScaredTimer(successor) > 0 else 0.0
    
    features['enemyPacmanNearMe'] = 0.0
    minOppDist = 10000
    minOppPos = (0, 0)
    for ep in self.teamData.opponentPositions:
      # For a feature later on
      if ep is not None and self.getMazeDistance(ep, position) < minOppDist:
        minOppDist = self.getMazeDistance(ep, position)
        minOppPos = ep
      if ep is not None and self.getMazeDistance(ep, position) <= 1 and self.isPositionInTeamTerritory(successor, ep):
        features['enemyPacmanNearMe'] = 1.0
        
    features['numSameFriends'] = 0
    for friend in self.getTeam(successor):
      if successor.getAgentState(self.index).isPacman is self.isPacman(successor):
        features['numSameFriends'] = features['numSameFriends'] + 1

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDiffDistance = min([1000] + [self.getMazeDistance(position, food) - self.getMazeDistance(minOppPos, food) for food in foodList if minOppDist < 1000])
      features['blockableFood'] = 1.0 if minDiffDistance < 1.0 else 0.0

    return features

  def getLegalNeighbors(self, position, walls):
    x,y = position
    x_int, y_int = int(x + 0.5), int(y + 0.5)
    neighbors = []
    for dir, vec in Actions._directionsAsList:
      dx, dy = vec
      next_x = x_int + dx
      if next_x < 0 or next_x == walls.width: continue
      next_y = y_int + dy
      if next_y < 0 or next_y == walls.height: continue
      if not walls.data[next_x][next_y]: neighbors.append((next_x, next_y))
    return neighbors

class GoalBasedAgent(GreatAgent):
  def chooseAction(self, gameState):
    if not self.firstTurnComplete:
      self.firstTurnComplete = True
      self.startingFood = len(self.getFoodYouAreDefending(gameState).asList())
      self.theirStartingFood = len(self.getFood(gameState).asList())
    
    self.updateBeliefs(gameState)
    
    # check if you happen to be a ghost and adjacent to an enemy then eat him
    actions = gameState.getLegalActions(self.index)
    for a in actions:
      nextAgentState = gameState.generateSuccessor(self.index, a).getAgentState(self.index)
      if (not nextAgentState.isPacman) and (nextAgentState.scaredTimer == 0):
        nextPosition = self.getSuccessor(gameState, a).getAgentPosition(self.index)
        if nextPosition in self.teamData.opponentPositions:
          return a
    
    self.threateningEnemyPositions = []
    opponentIndices = self.getOpponents(gameState)
    for i, position in enumerate(self.teamData.opponentPositions):
      opponentState = gameState.getAgentState(opponentIndices[i])
      if (not opponentState.isPacman) and (opponentState.scaredTimer == 0):
        self.threateningEnemyPositions.append(position)

    # if we don't have a goal or we have already reached our goal then pick a new one
    if self.teamData.goal == None or self.getPosition(gameState) == self.teamData.goal:
      self.pickNewGoal(gameState)
    
    nextAction = self.nextActionForGoal(gameState, self.teamData.goal)
    
    # if our current goal is going to get us killed pick a new goal
    if self.actionWillGetYouEaten(gameState, nextAction):
      self.pickNewGoal(gameState)
      nextAction = self.nextActionForGoal(gameState, self.teamData.goal)

    return nextAction
    
  def nextActionForGoal(self, gameState, goal):
    actions = gameState.getLegalActions(self.index)    
    
    values = []
    for a in actions:
      if a == Directions.STOP: 
        values.append(10000) #don't stop
        continue
      nextPosition = self.getSuccessor(gameState, a).getAgentPosition(self.index)
      values.append(self.getMazeDistance(goal, nextPosition))
    
    minValue = min(values)
    possibleActions = [a for a, v in zip(actions, values) if v == minValue]
    
    return self.pickActionFurthestFromEnemies(gameState, possibleActions)
  
  def actionWillGetYouEaten(self, gameState, action):
    # update this to take dead-ends into account
    currentPosition = self.getPosition(gameState)
    currentPositionDistance = self.distanceToClosestEnemy(gameState, currentPosition)
    nextAgentState = gameState.generateSuccessor(self.index, action).getAgentState(self.index)
    nextPosition = self.getSuccessor(gameState, action).getAgentPosition(self.index)
    nextPositionDistance = self.distanceToClosestEnemy(gameState, nextPosition)
    
    # you won't get eaten if you are a ghost and not scared
    if (not nextAgentState.isPacman) and (nextAgentState.scaredTimer == 0):
      return False
    
    # don't go into a dead-end if it's going to kill you
    # disable this check to make the agent suicide for food in dead-ends
    # perhaps adjust it dynamically based on how the game is going
    if (currentPositionDistance - 1) < (2 * self.teamData.deadEnds[nextPosition]):
      return True
    
    # adjust this to control how close the agent is willing to get to an enemy (min value is 2)
    return nextPositionDistance < 4 and nextPositionDistance < currentPositionDistance 
    
  # returns the closest food that will not send you towards an enemy agent
  # if it can't find one then just set the goal to run away from the closest enemy
  def pickNewGoal(self, gameState):
    currentPosition = self.getPosition(gameState)
    remainingFoods = self.getFood(gameState).asList()
    
    # list of (distance to food, food location)
    values = []
    for food in remainingFoods:
      values.append((self.getMazeDistance(currentPosition, food), food))
    
    values.sort(key=lambda x: x[0])
    
    for distance,food in values:
      if (not self.actionWillGetYouEaten(gameState, self.nextActionForGoal(gameState, food))):
        self.teamData.goal = food
        return
    
    action = self.pickActionFurthestFromEnemies(gameState, gameState.getLegalActions(self.index))
    self.teamData.goal = self.getSuccessor(gameState, action).getAgentPosition(self.index)
    
  def pickActionFurthestFromEnemies(self, gameState, possibleActions):
    #list of (distanceToEnemy, action)
    values = []
    currentPosition = gameState.getAgentPosition(self.index)
    for action in possibleActions:
      nextPosition = self.getSuccessor(gameState, action).getAgentPosition(self.index)
      if action == Directions.STOP: 
        values.append((-2,action)) #give stopping a negative value
        continue
      if self.teamData.deadEnds[nextPosition] > self.teamData.deadEnds[currentPosition]:
        values.append((-1,action)) #give going further into a dead end a negative value, but not as negative as stopping
        continue
      nextAgentState = gameState.generateSuccessor(self.index, action).getAgentState(self.index)
      if (not nextAgentState.isPacman) and (nextAgentState.scaredTimer == 0):
        values.append((10000,action)) # give a very high value to going back to safe side
        continue
      values.append((self.distanceToClosestEnemy(gameState, nextPosition),action))
    
    values.sort(key=lambda x: x[0], reverse=True)
    maxDist = values[0][0]
    possibleActions = [action for dist, action in values if dist == maxDist]
    return random.choice(possibleActions)
  
  def distanceToClosestEnemy(self, gameState, currentLocation):
    values = []
    for enemy in self.threateningEnemyPositions:
      values.append(self.getMazeDistance(enemy, currentLocation))
    if len(values) == 0:
      return 10000
    return min(values)

class ExperimentalAgent(GreatAgent):
 
  def getFeatures(self, gameState, action):
    successor= self.getSuccessor(gameState, action)
    nextPosition = successor.getAgentPosition(self.index)
    features = util.Counter()
    
    foodList = self.getFood(successor).asList()
    foodScore = 0
    for food in foodList:
      foodScore += 1.0 / self.getMazeDistance(nextPosition, food)
    features['foodProximity'] = foodScore 
    
    features['foodLeft'] = len(foodList)
    
    threateningEnemyPositions = []
    opponentIndices = self.getOpponents(successor)
    for i, position in enumerate(self.teamData.opponentPositions):
      opponentState = successor.getAgentState(opponentIndices[i])
      if not opponentState.isPacman:
        threateningEnemyPositions.append(position)
    
    if len(threateningEnemyPositions) > 0:
      closestOpponentDistance = self.getMazeDistance(threateningEnemyPositions[0], nextPosition)
      for p in threateningEnemyPositions[1:]:
        opponentDistance = self.getMazeDistance(p, nextPosition)
        if opponentDistance < closestOpponentDistance:
          closestOpponentDistance = opponentDistance
      features['closestGhostDistanceInverse'] = 1.0 / (closestOpponentDistance + 1)
    
    features['inADeadEndWithGhostNearby'] = self.teamData.deadEnds[nextPosition] * features['closestGhostDistanceInverse']
    
    #if self.index == 1:
      #print nextPosition
      #print "Food Proximity:", foodScore
      #print "closestGhostDistanceInverse", features['closestGhostDistanceInverse']
      #print threateningEnemyPositions
      #print self.teamData.opponentPositions
          
    return features

  def getWeights(self, gameState, action):
    successorState = self.getSuccessor(gameState, action)
    weights = util.Counter()
    weights['foodProximity'] = 1
    weights['foodLeft'] = -5
    
    weights['closestGhostDistanceInverse'] = -5

    if successorState.getAgentState(self.index).isPacman:
      weights['inADeadEndWithGhostNearby'] = -1
    
    return weights


class OffensiveGreatAgent(GreatAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = self.getMutationFeatures(gameState, action)
    successor = self.getSuccessor(gameState, action)
    
    features['successorScore'] = self.getScore(successor)
    
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    features['numFood'] = len(foodList)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    weights = regularMutation.aggressiveDWeightsDict
    weights['successorScore'] = 1.5
    # Always eat nearby food
    weights['numFood'] = -1000
    # Favor reaching new food the most
    weights['distanceToFood'] = -5
    return weights

class DefensiveGreatAgent(GreatAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = self.getMutationFeatures(gameState, action)
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to all invaders
    invaderPositions = []
    opponentIndices = self.getOpponents(successor)
    for i, position in enumerate(self.teamData.opponentPositions):
      opponentState = successor.getAgentState(opponentIndices[i])
      if opponentState.isPacman:
        invaderPositions.append(position)
    
    features['numInvaders'] = len(invaderPositions)
    if len(invaderPositions) > 0:
      dists = [self.getMazeDistance(myPos, pos) for pos in invaderPositions]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    
    foodList = self.getFoodYouAreDefending(successor).asList()
    distance = 0
    for food in foodList:
      distance = distance + self.getMazeDistance(myPos, food)
    features['totalDistancesToFood'] = distance

    return features

  def getWeights(self, gameState, action):
    weights = regularMutation.goalieDWeightsDict
    weights['numInvaders'] = -100
    weights['onDefense'] = 100
    weights['invaderDistance'] = -1.5
    weights['totalDistancesToFood'] = -0.1
    weights['stop'] = -1
    weights['reverse'] = -1
    return weights


