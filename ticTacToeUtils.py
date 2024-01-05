######################################################################
# ticTacToeUtils.py                                                  #
# Author:  Dario Ghersi                                              #
# Version: 20240103                                                  #
# Notes:   Parts of this code were translated from my original       #
#          Racket code by GTP-4                                      #
######################################################################

def checkState(b):
    """
    Check whether one of the two players has won the game
    """
    if won(b, 'X'):
        return -10
    elif won(b, 'O'):
        return 10
    elif '-' not in b:
        return 0
    else:
        return None

#########################################################

def evaluateMove(b, x):
    return minimax(getGameTree(b[:x] + ['O'] +
                                 b[x+1:], ['X', 'O']),
                   ['min', 'max'])

#########################################################

def findEmptyCells(b):
    return [i for i, cell in enumerate(b) if cell == '-']

#########################################################

def getGameTree(b, players):
    state = checkState(b)
    if state is not None:
        return state
    return [getGameTree(b[:i] + [players[0]]
                          + b[i+1:], players[::-1])
            for i in findEmptyCells(b)]

#########################################################

def minimax(gameTree, maxMinL):
    if not isinstance(gameTree, list):
        return gameTree
    scores = [minimax(x, maxMinL[::-1])
              for x in gameTree]
    
    if maxMinL[0] == 'max':
        return max(scores)
    else:
        return min(scores)

#########################################################

def playAgainstAI():
    b = ['-' for _ in range(9)]
    allowed = list(range(9))

    printBoard(b)

    while checkState(b) is None:
        
        # Human move
        humanMove = int(input("Your move (0-8): "))
        while humanMove not in allowed or b[humanMove] != '-':
            humanMove = int(input("Invalid move. Try again (0-8): "))
        b[humanMove] = 'X'
        allowed.remove(humanMove)
        printBoard(b)
        if checkState(b) is not None:
            break

        # AI move
        bestScore = -float('inf')
        aiMove = None
        for x in allowed:
            score = evaluateMove(b, x)
            if score > bestScore:
                bestScore = score
                aiMove = x
        b[aiMove] = 'O'
        allowed.remove(aiMove)
        printBoard(b)

    result = checkState(b)
    if result == 10:
        print("AI wins!")
    elif result == -10:
        print("You win!")
    else:
        print("It's a tie!")

#########################################################

def printBoard(b):
    """
    Format the board list as a 3x3 table
    """
    print("\n {} {} {}   0 1 2\n {} {} {}   3 4 5\n {} {} {}   6 7 8\n".format(*b))

#########################################################

def subseq(lst, start, end):
    return lst[start:end]

#########################################################

def won(b, player):
    lines = [b[0:3], b[3:6], b[6:9], b[0:7:3], b[1:8:3],
             b[2:9:3], b[0:9:4], b[2:7:2]]
    return any(all(cell == player for cell in line)
               for line in lines)
