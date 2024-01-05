# ticTacToe
Tic-Tac-Toe game implementation in Python, featuring an AI component for game strategy (using minimax) and a Genetic Algorithm (GA) implementation for optimizing these strategies. It consists of three main files:

* `ticTacToeAI.py`: Contains the main game logic and AI strategies for playing Tic-Tac-Toe.
* `ticTacToeGA.py`: Applies Genetic Algorithms to optimize Tic-Tac-Toe playing strategies.
* `ticTacToeUtils.py`: Provides utility functions supporting the game and algorithm implementations.

The program can be used to illustrate how GA can evolve a strategy that is competitive with the optimimal one obtained with minimax. There are several functions to evaluate strategies. The one used by default plays against all possible moves and returns the fractions of games won or tied, as described in [Hochmuth's whitepaper](https://citeseerx.ist.psu.edu/doc_view/pid/dc034fd0f5819867324fe1ab0d721262b97b7704).

You can also play against minimax using the `ticTacToeAI.py` script.
