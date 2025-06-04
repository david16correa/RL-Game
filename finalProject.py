import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

# memoization
rfCache = []
tfCache = []

def regenCache():
    global rfCache
    global tfCache

    rfCache = []
    tfCache = []

# =============================================================================================
# =============================================================================================
# ----------------------------------------- game setup ----------------------------------------
# =============================================================================================
# =============================================================================================

# player-controlled character in this game; it has a position, a reward, and it knows if it's
# reached the goal.
class Walker:
    def __init__(self, position=None, reward=None, pickedPackages = None, goalReached=None):
        # default values
        if position==None:
            position=[0,0]
        if reward==None:
            reward=0
        if pickedPackages==None:
            pickedPackages=[]
        if goalReached==None:
            goalReached=False
        self.position = np.array(position)
        self.reward = reward
        self.pickedPackages = pickedPackages
        self.goalReached = goalReached

# valid moves (action space)
directions = {
    "left":  [-1,0],
    "down":  [0,-1],
    "up":    [0,1],
    "right": [1,0],
}

actions = ['left','down','up','right']

# the labyrinth of the game; it's defined as a sqare grid of side length `size`,
# where hallways are individually listed
class Maze:
    def __init__(self, size=None, hallways=None):
        # default values
        if size==None:
            size=10

        self.size = size
        self.plan = np.asarray([[i * size + j + 1 for j in range(size)] for i in range(size)])

        # maze generator: hallways are randomly added, one by one, until all rooms are connected
        if hallways==None:
            hallways = []
            # for every room (node) in the maze, at least one hallway connecting it to a distinct room
            # is chosen. The number of hallways must then be at least enough to connect all rooms.
            while not mazeIsGood(hallways, size):
                # indices are used to choose a random room
                i,j = random.choices(range(size), k = 2)
                ogRoom = self.plan[i,j]
                # a random neighboring room is chosen to form a hallway
                targetRoom = ogRoom
                while ogRoom == targetRoom:
                    # a random direction is chosen (up, down, left right) for the proposed hallway
                    hallway = random.choice(list(directions.values()))
                    # if the hallway leads outside the maze (out of bounds) it is brought back.
                    # This might lead into the target room being equal to the original room. Hence
                    # the while loop.
                    Id = max(0, min(i+hallway[0], self.size-1))
                    Jd = max(0, min(j+hallway[1], self.size-1))
                    targetRoom = self.plan[Id, Jd]
                # once the hallway is chosen, it is saved to the list of hallways as an ordered tuple.
                # This allows redundant hallways to be ignored.
                hallways.append(tuple(np.sort([ogRoom, targetRoom])))
                hallways = list(set(hallways))

            # once the hallways are chosen, they're saved
            self.hallways = set(hallways)

# some object; it can be the goal, or just some package the walker must pickup.
# it has a position and some value
class Box:
    def __init__(self, position=None, value=None):
        # default values
        if position==None:
            position=[0,0]
        if value==None:
            value=10
        self.position = np.array(position)
        self.value = value

# the game itself; it has the maze, a walker, the goal, some packages, and a clock. This
# class also includes the following methods:
#    - moveWalker(direction): will move the walker in a given direction
#    - plot(): will return `fig` and `ax` as the graphical representation of the game
#              at the current time
#    - copy(): will return a deep copy of the game
class Game:
    def __init__(self,
        size=None,
        hallways=None,
        goal=None,
        walker=None,
        nPackages = None,
        goalValue=5,
        packageValue=15,
        timePenalty=1,
        invalidMovePenalty=5
    ):
        regenCache()

        # default values
        if walker==None:
            walker=Walker()
        if nPackages == None:
            nPackages = 3
        if goal == None:
            goal = [size-1,size-1]

        self.maze = Maze(size,hallways)
        self.walker = walker
        self.goal = Box(goal, value=goalValue)
        self.time = 0
        self.timePenalty = timePenalty
        self.invalidMovePenalty = invalidMovePenalty

        # the desired number of packages are generated ensuring their positions do not coincide
        # with the player, the goal, or themselves.
        self.packages = []
        boxPositions = [tuple(self.walker.position), tuple(self.goal.position)]
        for Id in range(nPackages):
            pos = tuple(self.walker.position)
            while (tuple(pos) in boxPositions):
                pos = [random.choice(range(self.maze.size)) for _ in range(2)]
            # once a distinct position is found, a box is created with some random reward, and saved
            boxPositions.append(tuple(pos))
            self.packages.append(Box(position = pos, value = packageValue))
            self.walker.pickedPackages.append(False)

    # moveWalker(direction): will move the walker in a given direction
    def moveWalker(self, direction):
        # if a string is chosen (which is the intended use), the corresponding direction is
        # retreived
        if type(direction) == str:
            direction = directions[direction.lower()]

        # the current room is retrieved
        ogPosition = self.walker.position.copy()
        ogRoom = self.maze.plan[*self.walker.position]
        # the walker is moved; if it bumps into a wall, it'll be returned to its original position
        self.walker.position += np.array(direction)
        self.walker.position[0] = max(0, min(self.walker.position[0], self.maze.size-1))
        self.walker.position[1] = max(0, min(self.walker.position[1], self.maze.size-1))

        # the resulting room after the displacement is saved
        targetRoom = self.maze.plan[*self.walker.position]

        # if the displacement does not correspond to a hallway, the player is penalized and sent back
        # to its original room; this includes both moving to a room that's disconnected to the original 
        # room, or bumping into a wall.
        if not(tuple(np.sort([ogRoom, targetRoom])) in self.maze.hallways):
            self.walker.position = ogPosition
            self.walker.reward -= self.invalidMovePenalty

        # if the new position coincides with a box that's not yet been picked up, its value is added as a reward
        for id in range(len(self.packages)):
            box = self.packages[id]
            if not self.walker.pickedPackages[id] and np.sum(np.abs(self.walker.position - box.position)) == 0:
                self.walker.reward += box.value
                self.walker.pickedPackages[id] = True

        # if the new position coincides with the goal, the player is rewarded and the game is over.
        # The player is told they've reached the goal (walker.goalReached = True)
        if np.sum(np.abs(self.walker.position - self.goal.position)) == 0:
            self.walker.reward += self.goal.value
            self.walker.goalReached = True

        # the time is updated, and the player is punished for it
        self.time += 1
        self.walker.reward -= self.timePenalty

    # plot(): will return `fig` and `ax` as the graphical representation of the game
    def plot(self):
        fig, axes = plt.subplots(figsize = (4.5,4.5))

        # all hallways are plot in a nice purple color
        for hallway in self.maze.hallways:
            xs = []
            ys = []
            aux = np.where(self.maze.plan == hallway[0])
            coordinate = tuple(zip(*aux))[0]
            xs.append(coordinate[0])
            ys.append(coordinate[1])

            aux = np.where(self.maze.plan == hallway[1])
            coordinate = tuple(zip(*aux))[0]
            xs.append(coordinate[0])
            ys.append(coordinate[1])

            axes.plot(xs, ys, linewidth=10, color='#DF87FF', zorder=1)  # 'o' marks endpoints

        # all packages that still hold their value are plot
        for id, box in enumerate(self.packages):
            if not self.walker.pickedPackages[id]:
                axes.scatter(box.position[0], box.position[1], s=50, color='red', zorder=2)

        # the goal and finally the player are plot
        axes.scatter(self.goal.position[0], self.goal.position[0], s=50, color='green', zorder=2)
        axes.scatter(self.walker.position[0], self.walker.position[1], s=50, color='blue', zorder=100)

        # Add a text box to the right of the plot
        axes.text(
            0.02, 1.05,
            f'Time:  {self.time}, '+
            f'Reward:  {self.walker.reward:.0f}',
            transform=axes.transAxes,
            fontsize=12,
            va='center',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        return fig, axes

    def copy(self):
        return copy.deepcopy(self)

# =============================================================================================
# =============================================================================================
# -------------------------------- Reinforcement learning: aux --------------------------------
# =============================================================================================
# =============================================================================================

# Count of all possible states
def stateSpaceSize(game):
    return game.maze.size**2 * 2**len(game.walker.pickedPackages)

# `stateId` assigns a single integer to all states of the form:
#    (x, y, p0, p1, p2, ...)   
# where pn indicates whether the nth package has been picked up or not.
# With this function, the entire state space is now indexed. This will
# be used in the value function.
# Its value is chosen to avoid redundancies; values go strictly from `0`
# to `game.maze.Size^2 * 2^len(game.walker.pickedPackages) - 1`
def stateId(game):
    x,y = game.walker.position
    positionId = (x*game.maze.size + y) * 2**len(game.walker.pickedPackages)
    cargoId = sum(2**Id for Id, picked in enumerate(game.walker.pickedPackages) if picked)

    return int(positionId + cargoId)

# computes the game as it would be if some action is taken; it returns the game id.
def transitionFunctionId(game, action):
    global tfCache
    # cache init
    if len(tfCache) == 0:
        tfCache = [[None for _ in range(len(actions))] for _ in range(stateSpaceSize(game))]

    # aux ids
    gameId = stateId(game)
    actionId = actions.index(action)

    # if necessary, generate cache
    if tfCache[gameId][actionId] == None:
        auxGame = game.copy()
        auxGame.moveWalker(action)
        tfCache[gameId][actionId] = stateId(auxGame)

    # return cache
    return tfCache[gameId][actionId]

# returns the reward of taking some action, given a game
def rewardFunction(game, action):
    global rfCache
    # cache init
    if len(rfCache) == 0:
        rfCache = [[None for _ in range(len(actions))] for _ in range(stateSpaceSize(game))]

    # aux ids
    gameId = stateId(game)
    actionId = actions.index(action)

    # if necessary, generate cache
    if rfCache[gameId][actionId] == None:
        auxGame = game.copy()
        auxGame.moveWalker(action)
        rfCache[gameId][actionId] = auxGame.walker.reward

    # return cache
    return rfCache[gameId][actionId]

# =============================================================================================
# =============================================================================================
# ------------------- Reinforcement learning: determining the optimal policy ------------------
# =============================================================================================
# =============================================================================================

def optimalPolicyGet(game, gamma = 0.5, policy = None, valueFunction=None, order=np.inf):
    # memoization - all possible states games
    gamesCache = [None] * stateSpaceSize(game)
    for x in range(game.maze.size):
        for y in range(game.maze.size):
            for packageOccupationBin in range(2**len(game.walker.pickedPackages)):
                auxGame = game.copy()
                auxGame.walker.position = [x,y]
                bits = bin(packageOccupationBin)[2:].zfill(len(game.walker.pickedPackages))
                auxGame.walker.pickedPackages = [bit == '1' for bit in bits]
                gamesCache[stateId(auxGame)] = auxGame

    # 0. initialization: value function and policy
    if policy==None:
        policy = [random.choice(actions) for _ in range(stateSpaceSize(game))]
    if valueFunction==None:
        valueFunction = [0] * stateSpaceSize(game)

    # policy iteration scheme
    it = 0 # used to truncate to desired order
    threshold = 1e-10
    policyStable = False
    while not policyStable and it < order:
        it+=1
        # 1. policy evaluation
        delta = 1
        peIt = 0
        while delta>threshold:
            peIt += 1
            delta = 0
            # loop over all states
            for sId, auxGame in enumerate(gamesCache):
                # if the goal is reached, the game is over, and the value function is zero
                if np.sum(np.abs(auxGame.walker.position - auxGame.goal.position)) == 0:
                    valueFunction[sId] = 0
                    continue

                oldVf = valueFunction[sId]
                action = policy[sId]
                # V(s) = R(s, π(s)) + γ * V(s')
                valueFunction[sId] = rewardFunction(auxGame, action) + gamma * valueFunction[transitionFunctionId(auxGame, action)]

                # delta is updated
                delta = max(delta, abs(oldVf - valueFunction[sId]))
        if order != np.inf:
            print(f"policy evaluation iterations used until convergence (delta <= threshold): {peIt}")

        # 2. policy improvement
        policyStable = True
        # loop over all states
        for sId, auxGame in enumerate(gamesCache):
            oldAction = policy[sId]
            # π(s) = argmax_a [ R(s, a) + γ * V(s') ]
            Vf = -np.inf
            for action in actions:
                candidateVf = rewardFunction(auxGame, action) + gamma * valueFunction[transitionFunctionId(auxGame,action)]
                if candidateVf >= Vf:
                    Vf = candidateVf
                    policy[sId] = action

            if policy[sId] != oldAction:
                policyStable = False

    # if the user requested some number of iterations, they're notified when stability has been reached
    if policyStable and it >= order:
        print("policy stability achieved!")

    return policy, valueFunction

# =============================================================================================
# =============================================================================================
# --------------------------- Reinforcement learning: visualization ---------------------------
# =============================================================================================
# =============================================================================================

def valueFunctionArray(game,X,Y,pickedPackages,valueFunction,policy):
    M = np.asarray([[None for x in X] for y in Y])
    for idx, x in enumerate(X):
        for idy, y in enumerate(Y):
            positionId = (np.round(x)*game.maze.size + np.round(y)) * 2**len(pickedPackages)
            cargoId = sum(2**Id for Id, picked in enumerate(pickedPackages) if picked)
            gameId = int(positionId + cargoId)
            M[idx,idy] = valueFunction[gameId]
    return np.array(M, dtype=float)

def policyArrays(game,X,Y,pickedPackages,valueFunction,policy):
    U = np.asarray([[None for x in X] for y in Y])
    V = np.asarray([[None for x in X] for y in Y])
    for idx, x in enumerate(X):
        for idy, y in enumerate(Y):
            positionId = (x*game.maze.size + y) * 2**len(pickedPackages)
            cargoId = sum(2**Id for Id, picked in enumerate(pickedPackages) if picked)
            gameId = int(positionId + cargoId)
            vec = directions[policy[gameId]]
            U[idx,idy] = vec[0]
            V[idx,idy] = vec[1]
    return (np.array(U, dtype=float),np.array(V, dtype=float))

# a moving averages routine is used to blur the an image
def blur(ogM, order=1):
    M = ogM.copy()
    sizeX, sizeY = M.shape
    for _ in range(order):
        for idx in np.arange(1,sizeX-1):
            for idy in np.arange(1,sizeY-1):
                M[idx,idy] += M[idx,idy+1] + M[idx,idy-1] + M[idx-1,idy] + M[idx+1,idy]
                M[idx,idy] += M[idx+1,idy+1] + M[idx+1,idy-1] + M[idx-1,idy+1] + M[idx-1,idy-1]
                M[idx,idy] /= 9

    return M

def RLplot(game,valueFunction,policy):
    length = 100
    x = np.asarray(np.linspace(-0.49, game.maze.size-0.51, num=length))
    quiverX = np.asarray(range(game.maze.size))
    quiverY = quiverX.copy()
    y = x.copy()

    vfM = valueFunctionArray(game,x,y,game.walker.pickedPackages,valueFunction,policy)
    policyU, policyV = policyArrays(game,quiverX,quiverY,game.walker.pickedPackages,valueFunction,policy)

    filtered = vfM[vfM != 0] # since the value function at the goal is defined as zero, zero is often an annoying outlier
    if len(filtered) != 0:
        vmin = filtered.min()
        vmax = filtered.max()
    else:
        vmin = 0
        vmax = 0

    fig, axes = game.plot()
    X, Y = np.meshgrid(x, y)
    cmap = plt.get_cmap('cividis')
    axes.pcolormesh(X, Y,
        blur(vfM, order=5).transpose(),
        vmin = vmin, vmax = vmax,
        cmap=cmap,#alpha = 0.85,
        zorder=-1,
    )
    axes.quiver(
        quiverX, quiverY,
        policyU.transpose(), policyV.transpose(),
        alpha=0.5
        # scale = 30,
    )
    epsilon = 0.25
    axes.set_xlim(0-epsilon, game.maze.size-1+epsilon)
    axes.set_ylim(0-epsilon, game.maze.size-1+epsilon)

    return fig, axes


# =============================================================================================
# =============================================================================================
# ---------------------------------- miscellaneous functions ----------------------------------
# =============================================================================================
# =============================================================================================

# a maze is good if all nodes are connected
def mazeIsGood(hallways, sideLength):
    # a spanning tree will be used; sets are used to avoid duplicates
    tree = {1}

    while True:
        # all neighbors to the positions contained in the tree are appended to the tree
        auxTree = list(tree)
        oldLength = len(tree)
        for pair in hallways:
            for leaf in tree:
                if (leaf in pair):
                    auxTree.append(int(pair[0]))
                    auxTree.append(int(pair[1]))
        tree = set(auxTree)

        # if the tree did not grow, it will no longer grow.
        if oldLength == len(tree):
            break

    # if the size of the tree is equal to the total number of nodes in the maze, then
    # it is connected
    return len(tree) == sideLength**2
