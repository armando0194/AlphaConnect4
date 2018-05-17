import numpy as np

class Board():
    def __init__(self, board=None, height=6, width=7, curr_player=1):
        ''' Initializes the Board and player '''
        self.height = height
        self.width = width
        self.board = np.zeros((self.height, self.width))
        self.curr_player = curr_player
    
    def make_move(self, col_idx):
        row_idx = np.where(self.board[:,col_idx] == 0)[0][-1]
        self.board[row_idx][col_idx] = self.curr_player
        self.curr_player *= -1
        return self.check_winner(row_idx, col_idx, self.curr_player*-1)
    
    def check_winner(self, row_idx, col_idx, player):
        row = self.board[row_idx, :]
        col = self.board[:, col_idx]
        diag1 = np.diagonal(self.board, col_idx - row_idx)
        diag2 = np.diagonal(np.fliplr(self.board), -(col_idx + row_idx - 6))

        print(row)
        print(col)
        print(diag1)
        print(diag2)
        # for line in [row, col, diag1, diag2]:
        #     if line.shape[0] < 4:
        #         continue

        #     for four in [line[i:i+4] for i in range(len(line)-3)]:
        #         if sum(four == 1) == 4:
        #             self.winner = self.player1
        #             return True
        #         elif sum(four == 2) == 4:
        #             self.winner = self.player2
        #             return True

        if sum(self.board[0] == 0) == 0:
            return True

        return False

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