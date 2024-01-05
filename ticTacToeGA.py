#!/usr/bin/python3

######################################################################
# ticTacToeGA.py                                                     #
# Author:  Dario Ghersi                                              #
# Version: 20240104                                                  #
# Notes:   Genetic Algorithms optimization of tic-tac-toe strategies #
######################################################################

from functools import partial
from multiprocessing import Pool
from ticTacToeUtils import *
from enumerateLegalBoardStates import MAP_TRANSFORMATION, transform,\
    generateTransformations
import numpy as np
import random
import sys

#random.seed(1) # debug -- fix the seed

######################################################################
# FUNCTIONS                                                          #
######################################################################

def convertToChromosomes(allS, uniquePositions):
    """
    Convert each strategy dictionary to a list of strings with the
    moves corresponding to the board states in uniquePositions
    """

    chromosomes = []
    for s in allS:
        chromosomes.append(list(map(lambda x: s[x], uniquePositions)))

    return chromosomes

######################################################################

def crossover(g1, g2, pCross):
    """
    Apply the crossover operator as described in Hochmuth's white
    paper (multiple cross sites).
    """

    newG1 = g1.copy()
    newG2 = g2.copy()

    for i in range(len(g1)):
        if random.random() < pCross: # swap the alleles
            temp = newG1[i]
            newG1[i] = newG2[i]
            newG2[i] = temp
    
    return newG1, newG2

######################################################################

def createRandomStrategy(positions):
    """
    Assign a random move to each of the 765 legal board states minus
    the ones that are a victory or a tie
    """
    
    strategy = {}
    for pos in positions:
        posL = pos.split(',')
        if not won(posL, 'X') and not won(posL, 'O') and\
           posL.count('-') > 0:

            # decide who is moving next
            symbol = 'O' if posL.count('O') < posL.count('X') else 'X'

            # pick a random empty position and place the symbol there
            available = [i for i, s in enumerate(posL) if s == '-']
            posL[random.choice(available)] = symbol

        strategy[pos] = ",".join(posL)

    return strategy

######################################################################

def evaluateStrategies(strategies, x):
    """
    Play strategy x against all other strategies, returning
    the fraction of games won or tied.
    'x' is the index of the strategy under evaluation
    """

    gamesNotLost = 0

    for i in range(len(strategies)):
        # other strategy first
        res = playGame(strategies[i], strategies[x])
        if res == 2 or res == 3:
            gamesNotLost += 1

        # strategy under evaluation first
        res = playGame(strategies[x], strategies[i])
        if res == 1 or res == 3:
            gamesNotLost += 1

    return gamesNotLost / (2 * len(strategies))

######################################################################

def evaluateStrategy(s):
    """
    Play the strategy against all possible moves, starting first
    and starting second, and returning the fraction of games that
    weren't lost (as discussed in Gregor Hochmuth's "On the Genetic
    Evolution of a Perfect Tic-Tac-Toe Strategy" whitepaper)
    """

    count = 0
    numGames = 0
    b = '-,' * 9
    b = b[:-1]
    
    ## strategy plays first
    res = evaluateStrategyR(s, b, stratTurn=True)
    res = flattenDeeplyNestedList(res)
    numGames += len(res)
    count += res.count(-10) + res.count(0)
    
    ## opponent plays first
    res = evaluateStrategyR(s, b, stratTurn=False)
    res = flattenDeeplyNestedList(res)
    numGames += len(res)
    count += res.count(10) + res.count(0)

    return count / numGames
    
######################################################################

def evaluateStrategyR(s, board, stratTurn):
    """
    Recursive function to play the strategy against all possible
    moves
    """

    #printBoard(board)
    if isinstance(board, str):
        board = board.split(",")

    # Check if the game has reached a terminal state
    state = checkState(board)
    if state is not None:
        return state

    if stratTurn:  # If it's 'X's turn (the strategy's turn)
        newBoard = board[:]
        newBoard = getNextMove(",".join(newBoard), s).split(",")
        return evaluateStrategyR(s, newBoard, False)
    
    else:  # If it's the opponent's turn
        outcomes = []
        available = [i for i, value in enumerate(board) if value == "-"]
        symbol = 'O' if board.count('O') < board.count('X') else 'X'
        for move in available:
            # Make a move
            newBoard = board[:]
            newBoard[move] = symbol

            # Recursive step
            outcome = evaluateStrategyR(s, newBoard, True)
            outcomes.append(outcome)

        return outcomes

######################################################################

def flattenDeeplyNestedList(nestedList):
    flattenedList = []
    for element in nestedList:
        if isinstance(element, list):
            # Extend with the result of a recursive call
            flattenedList.extend(flattenDeeplyNestedList(element))
        else:
            # Append it directly to the flattened list
            flattenedList.append(element)
            
    return flattenedList

######################################################################

def getNextMove(b, s, printFlag=False):
    """
    Make the next move, using board tranformations if necessary
    """
    
    if b in s:
        b = s[b]
        if printFlag:
            printBoard(b.split(","))
    else:
        trans = generateTransformations(b.split(","))
        for i in range(len(trans)):
            t = ",".join(trans[i])
            if t in s:
                b = s[t]
                b = transform(b.split(","), MAP_TRANSFORMATION[i])
                if printFlag:
                    printBoard(b)
                b = ",".join(b)
                break

    return b

######################################################################

def getPerfectStrategy(uniquePositions):
    """
    Use minimax to obtain the perfect strategy
    """

    perfectStrategy = {}
    for pos in uniquePositions:
        b = pos.split(",")
        symbol = 'O' if b.count('O') < b.count('X') else 'X'
        available = [i for i, s in enumerate(b) if s == '-']

        aiMove = None
        bestScore = -float('inf')
        for x in available:
            if symbol == 'O':
                score = minimax(getGameTree(b[:x] + ['O'] +
                                            b[x+1:], ['X', 'O']),
                                ['min', 'max'])
            else:
                score = -minimax(getGameTree(b[:x] + ['X'] +
                                             b[x+1:], ['O', 'X']),
                                 ['max', 'min'])
            if score > bestScore:
                bestScore = score
                aiMove = x

        newBoard = b.copy()
        newBoard[aiMove] = symbol
        perfectStrategy[pos] = ",".join(newBoard)

    return perfectStrategy

######################################################################

def humanMove(b):

    printBoard(b)
    humanMove = int(input("Your move (0-8): "))
    allowed = [i for i, value in enumerate(b) if value == "-"]
    while humanMove not in allowed or b[humanMove] != '-':
        humanMove = int(input("Invalid move. Try again (0-8): "))

    return humanMove
        
######################################################################

def mutate(genome, uniquePositions, p):
    """
    Apply the mutation operator to a genome with probability p.
    If runif(1) < p, then a random change is applied. The mutate
    operator is applied to each gene in a genome (strategy).
    """

    ## scan each gene in the genome
    for pos in range(len(genome)):

        # apply the mutation operator, by randomly picking a
        # different move if possible
        if random.random() < p:
            # decide who is moving next
            posL = uniquePositions[pos].split(",")
            symbol = 'O' if posL.count('O') < posL.count('X') else 'X'

            # remove the index of the actual move
            available = [i for i, s in enumerate(genome[pos].split(","))
                                                 if s == '-']

            if len(available) > 0:
                genome[pos] = posL
                genome[pos][random.choice(available)] = symbol
                genome[pos] = ",".join(genome[pos])

    return genome

######################################################################

def parallelEvaluateChromosomes(chromosomes, uniquePos,
                            popSize, numProcesses=None):

    ## combine chromosomes and uniquePos in a dictionary
    strategies = list(map(lambda x: dict(zip(uniquePos, chromosomes[x])),
                     range(len(chromosomes))))

    ## create a pool with the desired number of processes
    with Pool(processes=numProcesses) as pool:
        # map the helper function across the range of indices
        fitness = pool.map(evaluateStrategy, strategies[:popSize])

    return fitness

######################################################################

def playAgainstStrategy(s, moveFirst=False):
    """
    Function to let a human player play against a given strategy.
    If the 'moveFirst' strategy is set to True, the computer will
    move first
    """

    nextPlayer = ""
    b = ['-' for _ in range(9)]
    symbols = {}
    if moveFirst:
        symbols["human"] = "O"; symbols["computer"] = "X"
    else:
        symbols["human"] = "X"; symbols["computer"] = "O"

    ## first move
    if moveFirst:
        b = s[",".join(b)].split(",")
        nextPlayer = "human"
    else:
        b[humanMove(b)] = "X"
        nextPlayer = "computer"

    ## remaining moves
    while not won(b, "X") and not won(b, "O") and b.count("-") > 0:
        if nextPlayer == "human":
            b[humanMove(b)] = symbols["human"]
            printBoard(b)
            nextPlayer = "computer"
        elif nextPlayer == "computer":
            b = getNextMove(",".join(b), s, True).split(",")
            printBoard(b)
            nextPlayer = "human"

######################################################################

def playGame(s1, s2, printFlag = False):
    """
    Play a complete game pitting strategy s1 against strategy s2
    and return 1 if s1 won, 2, if s2 won, and 3 if it was a tie
    """

    ## first move
    b = s1['-,-,-,-,-,-,-,-,-']

    ## remaining moves
    for i in range(8):
        if printFlag:
            print("Move: ", i + 1)
        s = s1 if i % 2 else s2
        b = getNextMove(b, s, printFlag)

        if won(b.split(","), "X"):
            return 1
        elif won(b.split(","), "O"):
            return 2

    ## if none of the strategies won, it's a tie
    return 3

######################################################################

def selectNewGeneration(chromosomes, fitness, uniquePos,
                        pCross, pMut):
    """
    Basic GA implementation (repeat until the new generation has the
    same size as the old:
    1. Select two parent chromosomes with a probability proportional
       to their fitness.
    2. Apply the crossover operator.
    3. Apply the mutation operator on the two offspring
    """

    newChromosomes = []
    targetSize = len(chromosomes)
    positions = list(range(targetSize))
    currSize = 0
    while currSize < targetSize:

        # select two new parents
        g1 = chromosomes[np.random.choice(positions, p=fitness)]
        g2 = chromosomes[np.random.choice(positions, p=fitness)]

        # apply the crossover operator
        newG1, newG2 = crossover(g1, g2, pCross)

        # apply the mutation operator
        newG1 = mutate(newG1, uniquePos, pMut)
        newG2 = mutate(newG2, uniquePos, pMut)

        newChromosomes.extend([newG1, newG2])
        currSize += 2

    return newChromosomes
    
    
######################################################################
# MAIN PROGRAM                                                       #
######################################################################

def main(infileName, numGen, popSize, pCross, pMut):
    """
    Main function
    """

    ## read the board mapping
    boardMapping = {}
    with open(infileName, "r") as infile:
        for line in infile:
            fields = line.split()
            b = fields[0].split(",")
            if won(b, "X") or won(b, "O") or b.count("-") == 0:
                continue
            boardMapping[fields[0]] = fields[1]

    ## generate random strategies
    uniquePositions = list(set(boardMapping.values()))
    uniquePositions.sort()
    strategies = []
    for i in range(popSize):
        strategies.append(createRandomStrategy(uniquePositions))

    ## convert the strategy dictionaries to strings of moves
    ## (using the uniquePositions order)
    chromosomes = convertToChromosomes(strategies, uniquePositions)

    ## get the perfect strategy
    #perfectStrategy = getPerfectStrategy(uniquePositions)
    
    ## evaluate the fitness of each strategy
    fitness = parallelEvaluateChromosomes(chromosomes,
                                          uniquePositions, popSize, 5)

    print("Initial max: ", np.max(fitness))
    ## normalize the fitness
    fitness = [f / sum(fitness) for f in fitness]\
        if sum(fitness) != 0 else [0 for _ in fitness]

    ## select, rinse, and repeat
    for i in range(numGen):
        # select a new generation
        chromosomes = selectNewGeneration(chromosomes, fitness,
                                          uniquePositions, pCross, pMut)
    
        ## evaluate the fitness of each strategy
        fitness = parallelEvaluateChromosomes(chromosomes,
                                              uniquePositions,
                                              popSize, 5)
        maxFitness = np.max(fitness)
        print("Gen.: ", i, maxFitness)
        if abs(maxFitness - 1.0) < 1E-6:
            break
        
        # normalize the fitness
        fitness = [f / sum(fitness) for f in fitness]\
            if sum(fitness) != 0 else [0 for _ in fitness]


######################################################################

if __name__ == "__main__":

    ## parse the parameters
    if len(sys.argv) != 6:
        print("Usage: ticTacToeGA.py BOARD_MAPPING_FILE POP_SIZE NUM_GEN P_CROSS P_MUT")
        sys.exit(1)
    boardMappingFileName = sys.argv[1]
    popSize = int(sys.argv[2])
    if popSize % 2 != 0:
        print("The population size should be an even number.")
        sys.exit(1)
    numGen = int(sys.argv[3])
    pCross = float(sys.argv[4])
    pMut = float(sys.argv[5])
    
    main(boardMappingFileName, numGen, popSize, pCross, pMut)
