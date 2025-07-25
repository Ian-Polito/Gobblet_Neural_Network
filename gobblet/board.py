from tkinter import *
import numpy as np

window=0
canvas=0

class GamePiece:
    def __init__(self, size, owner):
        self.size = size
        self.owner = owner
        
class Board:
    def __init__(self):
        self.grid = [[[] for _ in range(4)] for _ in range(4)]
        self.grid2 = [[[] for _ in range(4)] for _ in range(4)]
        self.external_stacks = {
            0: [ # player 0
                [GamePiece(size,0) for size in (range(1,5))] for _ in range(3)
            ],
            1: [ # player 1
                [GamePiece(size,1) for size in (range(1,5))] for _ in range(3)
            ]}
        self.external_stacks2 = {0:[None for _ in range(3)],1:[None for _ in range(3)]}
        self.current_player = np.random.randint(1)
        self.possible_moves = self.enumerate_all_possible_moves()
    
    # returns a list of all possible move tuples in a fixed order
    # each move is either:
    # ("external", stack_index, to_row, to_col)
    # ("board", from_row, from_col, to_row, to_col)
    def enumerate_all_possible_moves(self):
        moves = []
        # external stack moves: 3 stacks x 4x4 board = 48
        for stack_index in range(3):
            for to_row in range(4):
                for to_col in range(4):
                    moves.append(("external", stack_index, to_row, to_col))
        
        # board to board moves: 16 positions x 15 destinations = 240
        for from_row in range(4):
            for from_col in range(4):
                for to_row in range(4):
                    for to_col in range(4):
                        if from_row != to_row or from_col != to_col:
                            moves.append(("board", from_row, from_col, to_row, to_col))
        return moves
    
    # given an index from the neural network output, return the corresponding move tuple
    def decode_move(self, index):
        if (0 <= index < len(self.possible_moves)):
            return self.possible_moves[index]
        else:
            return IndexError(f"Invalid move index: {index}")
    
    # if moving from an external stack, ext_stack is the numbered external stack; from_row
    # and from_col won't be needed. if moving from a game board stack, ext_stack won't be
    # needed and from_col and from_row will be the coordinates on the game board
    def move_piece(self, ext_stack, from_row, from_col, to_row, to_col):
        self.grid2 = [[stack.copy() for stack in row] for row in self.grid]
        self.external_stacks2 = {player: [stack.copy() for stack in stacks] for player, stacks in self.external_stacks.items()}
        if ext_stack is not None:
            piece = self.external_stacks[self.current_player][ext_stack].pop()
            self.grid[to_row][to_col].append(piece)
        else:
            piece = self.grid[from_row][from_col].pop()
            self.grid[to_row][to_col].append(piece)
        self.current_player = 1 - self.current_player
            
    # from_stack will either be an int (indicating the numbered external stack) or a stack
    # on the game board. if from_stack is a game board stack, player won't be needed
    def is_valid_move(self, player, from_stack, to_row, to_col):
        if not (0 <= to_row < 4 and 0 <= to_col < 4):
            return False
        if isinstance(from_stack, int):
            if not self.external_stacks[player][from_stack]:
                return False
            if self.grid[to_row][to_col]:
                if not self.valid_external_gobble(to_row, to_col):
                    return False
            piece = self.external_stacks[player][from_stack][-1]
        else:
            if not from_stack:
                return False
            piece = from_stack[-1]
        if not self.grid[to_row][to_col] or self.grid[to_row][to_col][-1].size < piece.size:
            if piece.owner == self.current_player:
                return True
        return False
    
    def valid_external_gobble(self, to_row, to_col):
        count = 0
        opponent = 1 - self.current_player
        # check row
        for x in range(4):
            if self.grid[to_row][x]:
                piece = self.grid[to_row][x][-1]
                if piece.owner == opponent:
                    count += 1
        if (count == 3):
            return True
        count = 0
        # check column
        for x in range(4):
            if self.grid[x][to_col]:
                piece = self.grid[x][to_col][-1]
                if piece.owner == opponent:
                    count += 1
        if (count == 3):
            return True
        count = 0
        # check left diagonal
        if (to_row == to_col):
            for x in range(4):
                if self.grid[x][x]:
                    piece = self.grid[x][x][-1]
                    if piece.owner == opponent:
                        count += 1
        if (count == 3):
            return True
        count = 0
        # check right diagonal, honestly can't think of a better way to check if the target
        # board space is one of these four spaces
        if ((to_row == 0 and to_col == 3) or (to_row == 1 and to_col == 2) or (to_row == 2 and to_col == 1) or (to_row == 3 and to_col == 0)):
            for x in range(4):
                if self.grid[x][3-x]:
                    piece = self.grid[x][3-x][-1]
                    if piece.owner == opponent:
                        count += 1
        if (count == 3):
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
    
    # checks whether moving the piece on the board will uncover a win for their opponent
    # returns true if it does, or false if it doesn't or the move interrupts the four in a row
    def uncover_check(self, from_row, from_col, to_row, to_col):
        if self.grid[from_row][from_col]: # check if the stack is valid
            piece = self.grid[from_row][from_col].pop() # temporarily uncover the target piece
            opponent = 1 - self.current_player
            win = self.check_win() # check if moving this piece uncovers a four in a row
            if win == opponent: # if moving the piece uncovers a four in a row
                if all((self.is_valid_move(self.current_player, self.grid[from_row][from_col], to_row, to_col)) and from_row != to_row and from_col != to_col):
                # check if the move is valid, and not putting the piece back where it was
                    self.grid[to_row][to_col].append(piece) # temporarily move the piece to the destination
                    if self.check_win() == opponent: # check if this interrupted the four in a row
                        self.grid[to_row][to_col].pop() # it did not, remove the piece from the destination
                        self.grid[from_row][from_col].append(piece) # move the piece back
                        return True # moving the piece uncovers a win for the opponent. Current player loses
                    self.grid[to_row][to_col].pop() # move interrupted the four in a row
                    self.grid[from_row][from_col].append(piece) # move the piece back
                    return False # moving the piece does not uncover a win for the opponent. Game continues
                self.grid[from_row][from_col].append(piece) # move the piece back
                return True # destination was invalid, so move uncovers a win for the opponent. Current player loses
            self.grid[from_row][from_col].append(piece) # move the piece back
            return False # moving the piece does not uncover a win
        return False # stack is invalid
    
    def encode_board(self):
        encoded = []
        # encode the 4x4 board (16 stacks, each with up to 4 pieces)
        for row in self.grid:
            for stack in row:
                # encode up to 4 pieces per stack (top to bottom)
                for i in range(4):
                    if i < len(stack):
                        piece = stack[i]
                        size_encoding = [0] * 4
                        owner_encoding = [0] * 2
                        size_encoding[piece.size - 1] = 1
                        owner_encoding[piece.owner] = 1
                        encoded.extend(size_encoding + owner_encoding)
                    else:
                        # pad with zeroes if fewer than 4 pieces
                        encoded.extend([0] * 6)
        # encode the external stacks (3 stacks per player, each with up to 4 pieces)
        for player in [0,1]:
            for stack in self.external_stacks[player]:
                for i in range(4):
                    if i < len(stack):
                        piece = stack[i]
                        size_encoding = [0] * 4
                        owner_encoding = [0] * 2
                        size_encoding[piece.size - 1] = 1
                        owner_encoding[piece.owner] = 1
                        encoded.extend(size_encoding + owner_encoding)
                    else:
                        # pad with zeroes
                        encoded.extend([0] * 6)
        assert len(encoded) == 528, f"Encoded board length is {len(encoded)}, expected 528"
        return encoded
        
    def initialize_visualization(self):
        self.window = Tk()
        self.canvas = Canvas(self.window, width=960, height=540)
        self.window.title("Gobblet Board Visualization")
        window_width = 960
        window_height = 540
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
        # draw the before board
        for x in range(4):
            for y in range(4):
                self.canvas.create_rectangle(128+(x*64),128+(y*64),192+(x*64),192+(y*64), fill="white", outline="black", width=2, tags="square")
        # draw the after board
        for x in range(4):
            for y in range(4):
                self.canvas.create_rectangle(576+(x*64),128+(y*64),640+(x*64),192+(y*64), fill="white", outline="black", width=2, tags="square")
        
        #  text above the boards
        self.canvas.create_text(256, 104, text="Current", fill="black", font=("Arial", 12, "bold"), tags="text")
        self.canvas.create_text(704, 104, text="Before", fill="black", font=("Arial", 12, "bold"), tags="text")
        
        # finally draw the external stacks
        for x in range(3):
            self.canvas.create_oval(48,160+(x*80),112,224+(x*80), fill="red", outline="black", width=1, tags="piece")
            self.canvas.create_text(((48+112)/2), (((160+224)/2)+(x*80)), text="4", fill="black", font=("Arial", 12), tags="number")
        for x in range(3):
            self.canvas.create_oval(400,128+(x*80),464,192+(x*80), fill="royalblue", outline="black", width=1, tag="piece")
            self.canvas.create_text(((400+464)/2), (((128+192)/2)+(x*80)), text="4", fill="black", font=("Arial", 12), tags="number")
            
        self.canvas.pack()
            
    def visualize_win(self, winner):
        if winner == 1 or winner == 0:
            self.canvas.create_text(480, 32, text=("Player", self.current_player, "wins."), fill="black", font=("Arial", 12, "bold"), tags="text")
        else:
            self.canvas.create_text(480, 32, text=("Game resulted in a Draw."), fill="black", font=("Arial", 12, "bold"), tags="text")
        
    def visualize_board_wrapper(self):
        self.window.mainloop()
        
    def game_loop(self, net1, net2, turn=0, max_turns=100):
        winner = self.check_win()
        if winner is not None or turn >= max_turns:
            self.visualize_win(winner)
            if winner is not None:
                print(f"Player {winner} wins!")
            else:
                print("Game ended in a draw.")
        current_net = [net1, net2][self.current_player]
        inputs = self.encode_board()
        outputs = current_net.activate(inputs)
        move_index = outputs.index(max(outputs))
        move = self.decode_move(move_index)
        
        valid = False
        if move[0] == "external":
            stack_index, to_row, to_col = move[1:]
            if self.is_valid_move(self.current_player, stack_index, to_row, to_col):
                self.move_piece(stack_index, None, None, to_row, to_col)
                valid = True
        elif move[0] == "board":
            from_row, from_col, to_row, to_col = move[1:]
            from_stack = self.grid[from_row][from_col]
            if self.is_valid_move(self.current_player, from_row, from_col, to_row, to_col):
                if self.uncover_check(from_row, from_col, to_row, to_col):
                    print(f"Player {1 - self.current_player} wins by uncovering!")
                    self.visualize_win(1 - self.current_player)
                    if from_row == to_row and from_col == to_col:
                        # for visual purposes, remove the piece to show the win
                        self.grid2 = self.grid.copy()
                        self.grid[from_row][from_col].pop()
                    else:
                        # for visual purposes, move the piece to show the win
                        self.move_piece(None, from_row, from_col, to_row, to_col)
                    self.visualize_board()
                    return
                else:
                    self.move_piece(None, from_row, from_col, to_row, to_col)
                    valid = True
                        
        if not valid:
            self.current_player = 1 - self.current_player
            print(f"Invalid move by player {self.current_player}")
        self.visualize_board()
        self.window.after(5000, lambda: self.game_loop(net1, net2, turn+1, max_turns)) # 5 second delay between moves
    
    def visualize_board(self):
        self.canvas.delete("piece")
        self.canvas.delete("number")
        
        # draw the current board state
        for x in range(4):
            for y in range(4):
                if self.grid[x][y]:
                    piece = self.grid[x][y][-1]
                    if piece.owner == 0:
                        color = "red"
                    else:
                        color = "royalblue"
                    self.canvas.create_oval(((128+(x*64))+((4-piece.size)*8)), ((128+(y*64))+((4-piece.size)*8)), (((128+(x*64))+((4-piece.size)*8))+(piece.size*16)), (((128+(y*64))+((4-piece.size)*8))+(piece.size*16)), fill=color, outline="black", width=1, tags="piece")
                    self.canvas.create_text((((128+(x*64))+((4-piece.size)*8))+(((128+(x*64))+((4-piece.size)*8))+(piece.size*16)))/2, (((128+(y*64))+((4-piece.size)*8))+(((128+(y*64))+((4-piece.size)*8))+(piece.size*16)))/2, text=str(piece.size), fill="black", font=("Arial", 12), tags="number")
        
        # draw the current external stacks
        for player in [0,1]:
            for y in [0,1,2]:
                if self.external_stacks[player][y]:
                    piece = self.external_stacks[player][y][-1]
                    if piece.owner == 0:
                        color = "red"
                        x=160
                    else:
                        color = "royalblue"
                        x=128
                    self.canvas.create_oval(((48+(player*352))+((4-piece.size)*8)), ((x+(y*80))+((4-piece.size)*8)), (((48+(player*352))+((4-piece.size)*8))+(piece.size*16)), (((x+(y*80))+((4-piece.size)*8))+(piece.size*16)), fill=color, outline="black", width=1, tags="piece")
                    self.canvas.create_text((((48+(player*352))+((4-piece.size)*8))+(((48+(player*352))+((4-piece.size)*8))+(piece.size*16)))/2, (((x+(y*80))+((4-piece.size)*8))+(((x+(y*80))+((4-piece.size)*8))+(piece.size*16)))/2, text=str(piece.size), fill="black", font=("Arial", 12), tags="number")
        
        # draw the previous board state
        for x in range(4):
            for y in range(4):
                if self.grid2[x][y]:
                    piece = self.grid2[x][y][-1]
                    if piece.owner == 0:
                        color = "red"
                    else:
                        color = "royalblue"
                    self.canvas.create_oval(((576+(x*64))+((4-piece.size)*8)), ((128+(y*64))+((4-piece.size)*8)), (((576+(x*64))+((4-piece.size)*8))+(piece.size*16)), (((128+(y*64))+((4-piece.size)*8))+(piece.size*16)), fill=color, outline="black", width=1, tags="piece")
                    self.canvas.create_text((((576+(x*64))+((4-piece.size)*8))+(((576+(x*64))+((4-piece.size)*8))+(piece.size*16)))/2, (((128+(y*64))+((4-piece.size)*8))+(((128+(y*64))+((4-piece.size)*8))+(piece.size*16)))/2, text=str(piece.size), fill="black", font=("Arial", 12), tags="number")
        
        # draw the previous external stacks
        for player in [0,1]:
            for y in [0,1,2]:
                if self.external_stacks2[player][y]:
                    piece = self.external_stacks2[player][y][-1]
                    if piece.owner == 0:
                        color = "red"
                        x=160
                    else:
                        color = "royalblue"
                        x=128
                    self.canvas.create_oval(((496+(player*352))+((4-piece.size)*8)), ((x+(y*80))+((4-piece.size)*8)), (((496+(player*352))+((4-piece.size)*8))+(piece.size*16)), (((x+(y*80))+((4-piece.size)*8))+(piece.size*16)), fill=color, outline="black", width=1, tags="piece")
                    self.canvas.create_text((((496+(player*352))+((4-piece.size)*8))+(((496+(player*352))+((4-piece.size)*8))+(piece.size*16)))/2, (((x+(y*80))+((4-piece.size)*8))+(((x+(y*80))+((4-piece.size)*8))+(piece.size*16)))/2, text=str(piece.size), fill="black", font=("Arial", 12), tags="number")
        self.canvas.pack()