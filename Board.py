import numpy as np

class Board():
    def __init__(self, board=None, height=6, width=7, curr_player=1):
        self.height = height
        self.width = width
        self.board = np.zeros((self.height, self.width))
        self.curr_player = 1
    
    def make_move(self, col):
        valid_moves = self.get_valid_moves()
        if valid_moves[col]:
            avail_rows = np.where(self.board[:,col] == 0)[0]
            self.board[avail_rows[-1]][col] = self.curr_player
            self.curr_player *= -1
        else:
            print('Invalid move')
    
    def check_winner():
        pass

    def get_valid_moves(self):
        return self.board[0] == 0

board = Board()
board.make_move(2)
print(board.board)
board.make_move(2)
print(board.board)
board.make_move(2)
print(board.board)
board.make_move(1)
print(board.board)
board.make_move(3)