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
    self.agents = ["experimental", "defense", "defense"]
  
  def getAgent(self, index):
      return self.choose(self.agents.pop(0), index)

  def choose(self, agentStr, index):
    if agentStr == 'offense':
      return OffensiveGreatAgent(index)
    elif agentStr == 'defense':
      return DefensiveGreatAgent(index)
    elif agentStr == 'experimental':
      return ExperimentalAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

##########
# Agents #
##########

class GreatAgent(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.firstTurnComplete = False
    self.startingFood = 0
    self.theirStartingFood = 0
  
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
    
    self.walls = gameState.getWalls()
    
    
    # Find all possible legal positions
    width = self.getLayoutWidth(gameState)
    height = self.getLayoutHeight(gameState)
    beliefs = util.Counter()
    legalPositions = []
    for i in range(0, width):
      for j in range(0, height):
        if gameState.hasWall(i,j) == False:
          legalPositions.append((i,j))
          beliefs[(i,j)] = 1
    beliefs.normalize()
                
    # set up beliefs for each opponent agent
    self.opponentBeliefs = []
    self.opponentPositions = self.getOpponentPositions(gameState)
    
    for opponentIndex in self.getOpponents(gameState):
      self.opponentBeliefs.append(beliefs.copy())
      
    # Find all dead ends in the maze
    numLegalNeighbors = util.Counter()
    possibleDeltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    self.deadEnds = util.Counter()
    frontier = []
    
    for p in legalPositions:
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
        if not gameState.hasWall(neighbor[0], neighbor[1]) and self.deadEnds[neighbor] == 0:
          numNonDeadEndNeighbors += 1
      if numNonDeadEndNeighbors == 1:
        self.increment(cur, util.Counter())
        self.deadEnds[cur] = 1
        for dx, dy in possibleDeltas:
          neighbor = (cur[0] + dx, cur[1] + dy)
          if not gameState.hasWall(neighbor[0], neighbor[1]) and self.deadEnds[neighbor] == 0:
            frontier.append(neighbor)

  def increment(self, position, alreadyVisited):
    if self.deadEnds[position] > 0:
      self.deadEnds[position] += 1
      alreadyVisited[position] = 1
    
    possibleDeltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in possibleDeltas:
      neighbor = (position[0] + dx, position[1] + dy)
      if self.deadEnds[neighbor] > 0 and alreadyVisited[neighbor] == 0:
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
    self.opponentPositions = self.getOpponentPositions(gameState)
    opponents = self.getOpponents(gameState)
    team = self.getTeam(gameState)
    teamPositions = self.getTeamPositions(gameState)
    
    for i, position in enumerate(self.opponentPositions):
      beliefs = self.opponentBeliefs[i]     
        
      # beliefs are re-initialized if pacman is captured
      if beliefs[beliefs.argMax()] == 0:
        beliefs.incrementAll(beliefs.keys(), 1)
        beliefs.normalize()
          
      if position == None:
        oldBeliefs = beliefs.copy()
        noisyDistance = gameState.getAgentDistance(opponents[i])
        for pos in beliefs:
          trueDistance = util.manhattanDistance(myPosition, pos)
          distanceProb = gameState.getDistanceProb(trueDistance, noisyDistance)
          if distanceProb == 0:
            beliefs[pos] = 0
          else:
            sum = 0
            previousPos = Actions.getLegalNeighbors(pos, self.walls)
            for prePos in previousPos:
              sum += oldBeliefs[prePos]
            beliefs[pos] += sum / len(previousPos) 
            beliefs[pos] *= distanceProb
        self.opponentPositions[i] = beliefs.argMax()
      else:
            for pos in beliefs:
                if beliefs[pos] > 0:
                    beliefs[pos] = 0
            beliefs[(position)] = 1
      beliefs.normalize()
    
    # print out opponents' positions (max prob)
    #if self.index == 0 or self.index == 1:
    #  self.displayDistributionsOverPositions(self.opponentBeliefs)
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
    for epos in self.opponentPositions:
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
    for ep in self.opponentPositions:
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

  def getFuturePositionsInEnemyArea(self, position, step):
      positions = []
      for i in range(step):
          neighbors = self.getLegalNeighbors(position, walls)
          
          
  def buildTreeFuturePositionsInTeamArea(self, gameState, depth):
      team = self.getTeam(gameState)
      teamPositions = self.getTeamPositions(gameState)
      positions = [[util.Counter() for i in range(len(team))] for j in range(depth)]
      positionTree = util.Counter()
      positionTreeNodeID = []
      positionTreeMax = util.Counter()
      
      index = 0
      for pos in teamPositions:
          if self.isPositionInTeamTerritory(gameState, pos) == True:
              positions[index][0][pos] = 0
              index += 1
      
      if index > 1:
          treeLevel = 0
          for dep in range(depth-1):
              for i in range(index):
                  for pos in positions[i][dep]:
                      for legalPos in self.getLegalNeighbors(pos, self.walls):
                          positions[i][dep+1][legalPos] = 0                  
                  self.addTreeNodes(positionTree, treeLevel, positions[i][dep+1])
                  treeLevel += 1
                  positionTreeNodeID.append(i)
      
      # find max distances at deepest nodes            
      #for dep in range(index - 1):
          
                  
      temp = 1
      

                  
      
                  
                  
  def addTreeNodes(self, tree, level, values):
      y = 0
      x = level
      while tree.has_key((x, y)):
        y += 1
      for val in values:
          x = level
          tree[(x,y)] = val
          y += 1


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
    for i, position in enumerate(self.opponentPositions):
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
    
    features['inADeadEndWithGhostNearby'] = self.deadEnds[nextPosition] * features['closestGhostDistanceInverse']
    
    #if self.index == 1:
     # print nextPosition
     # print "Food Proximity:", foodScore
     # print "closestGhostDistanceInverse", features['closestGhostDistanceInverse']
     # print threateningEnemyPositions
     
    self.buildTreeFuturePositionsInTeamArea(gameState, 3)  
          
    return features

  def getWeights(self, gameState, action):
    successorState = self.getSuccessor(gameState, action)
    weights = util.Counter()
    weights['foodProximity'] = 1
    weights['foodLeft'] = -5
    
    weights['closestGhostDistanceInverse'] = -3

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
    for i, position in enumerate(self.opponentPositions):
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


