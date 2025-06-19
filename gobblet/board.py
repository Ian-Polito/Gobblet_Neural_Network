import numpy as np

class GamePiece:
    def __init__(self, size, owner):
        self.size = size
        self.owner = owner
        
class Board:
    def __init__(self):
        self.grid = [[[] for _ in range(4)] for _ in range(4)]
        self.external_stacks = {0: {size: [GamePiece(size,0) for _ in range(3)] for size in range(1,5)},
                                1: {size: [GamePiece(size,1) for _ in range(3)] for size in range(1,5)}}
        self.current_player = 0
    
    def move_piece(self, from_stack, from_row, from_col, to_row, to_col, size=None):
        if from_stack:
            if not self.external_stacks[self.current_player][size]:
                return False
            piece = self.external_stacks[self.current_player][size].pop()
        else:
            if not self.grid[from_row][from_col]:
                return False
            piece = self.grid[from_row][from_col].pop()
        
        if self.is_valid_move(piece, to_row, to_col):
            self.grid[to_row][to_col].append(piece)
            self.current_player = 1 - self.current_player
            return True
        else:
            if from_stack:
                self.external_stacks[self.current_player][size].append(piece)
            else:
                self.grid[from_row][from_col].append(piece)
            return False
            
    def is_valid_move(self, piece, to_row, to_col):
        if not self.grid[to_row][to_col] or self.grid[to_row][to_col][-1].size < piece.size:
            return True
        return False
        
    def check_win(self):
        for player in [0,1]:
            for row in range(4):
                if all(self.grid[row][col] and self.grid[row][col][-1].owner == player for col in range(4)):
                    return player
            for col in range(4):
                if all(self.grid[row][col] and self.grid[row][col][-1].owner == player for row in range(4)):
                    return player
            if all(self.grid[i][i] and self.grid[i][i][-1].owner == player for i in range(4)):
                return player
            if all(self.grid[i][3-i] and self.grid[i][3-i][-1].owner == player for i in range(4)):
                return player
        return None
    
    def uncover_check(self, from_row, from_col, to_row, to_col):
        piece = self.grid[from_row][from_col].pop()
        opponent = 1 - self.current_player
        win = self.check_win()
        self.grid[from_row][from_col].append(piece)
        if win == opponent:
            if self.is_valid_move(piece, to_row, to_col):
                self.grid[to_row][to_col].append(piece)
                if self.check_win() == opponent:
                    self.grid[to_row][to_col].pop()
                    return False
                self.grid[to_row][to_col].pop()
            return False
        return True
    
    def encode_board(self):
        encoded = []
        for row in range(4):
            for col in range(4):
                stack = self.grid[row][col]
                for piece in stack:
                    size_one_hot = [0] * 4
                    size_one_hot[piece.size - 1] = 1
                    owner_one_hot = [0] * 2
                    owner_one_hot[piece.owner] = 1
                    encoded.extend(size_one_hot + owner_one_hot)
                encoded.extend([0] * (24 - len(stack) * 6))
        return encoded