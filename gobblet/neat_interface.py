import neat
import numpy as np
from .board import Board

def evaluate_genome(genome, config, opponents):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    
    for opponent_genome in opponents:
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
        fitness += play_game(net, opponent_net)
    genome.fitness = fitness / len(opponents)
    
def eval_genomes(genomes, config):
    for i, (genome_id, genome) in enumerate(genomes):
        opponents = [g for j, (gid, g) in enumerate(genomes) if j!= i][:3]
        evaluate_genome(genome, config, opponents)

def calculate_line_bonus(board, player, row, col):
    bonus = 0
    
    # Check all four directions: horizontal, vertical, diagonal, anti-diagonal
    directions = [
        [(0, 1), (0, -1)],    # horizontal
        [(1, 0), (-1, 0)],    # vertical  
        [(1, 1), (-1, -1)],   # diagonal
        [(1, -1), (-1, 1)]    # anti-diagonal
    ]
    
    for direction_pair in directions:
        count = 1  # Count the piece we just placed
        
        # Check both directions along this line
        for dr, dc in direction_pair:
            r, c = row + dr, col + dc
            while 0 <= r < 4 and 0 <= c < 4:
                if board.grid[r][c] and board.grid[r][c][-1].owner == player:
                    count += 1
                    r, c = r + dr, c + dc
                else:
                    break
        
        # Award bonuses for longer lines
        if count == 2:
            bonus += 10  # Two in a line
        elif count == 3:
            bonus += 30  # Three in a line
    
    return bonus
    
def play_game(net1, net2, max_turns=64):
    board = Board()
    players = [net1, net2]
    fitness = [0, 0]
    turn = 0
    # line bonus metric to keep track if a genome has already gotten the two in a row and three in a row bonuses
    line_bonus_awarded = [[False, False], [False, False]]  # [player][0=two_in_row, 1=three_in_row]
    
    while board.check_win() is None and turn < max_turns:
        current_net = players[board.current_player]
        inputs = board.encode_board()
        outputs = current_net.activate(inputs)
        # mask out invalid moves
        valid_mask = board.get_valid_move_mask()
        masked_outputs = [outputs[i] if valid_mask[i] else -float("inf") for i in range(len(outputs))]
        
        if all(v == -float("inf") for v in masked_outputs):
            break  # no valid moves, end as draw
        
        move_index = np.argmax(masked_outputs)
        move = board.decode_move(move_index)
        current_player = board.current_player  # capture before move_piece flips it
        
        if move[0] == "external":
            stack_index, to_row, to_col = move[1:]
            board.move_piece(stack_index, None, None, to_row, to_col)
                
        elif move[0] == "board":
            from_row, from_col, to_row, to_col = move[1:]
            if board.uncover_check(from_row, from_col, to_row, to_col):
                # move results in a loss due to uncovering a win without interrupting it
                fitness[1 - board.current_player] += 200
                fitness[board.current_player] -= 100 # penalty for uncovering opponent's win
                return fitness[0]
            else:
                board.move_piece(None, from_row, from_col, to_row, to_col)
        turn += 1
        
        # STRATEGIC BONUSES: Check for pieces in a line after placing
        line_bonus = calculate_line_bonus(board, current_player, to_row, to_col)
        if line_bonus > 0:
            if line_bonus >= 10 and not line_bonus_awarded[current_player][0]:
                fitness[current_player] += 10
                line_bonus_awarded[current_player][0] = True
            if line_bonus >= 30 and not line_bonus_awarded[current_player][1]:
                fitness[current_player] += 30
                line_bonus_awarded[current_player][1] = True
    
    # END OF GAME BONUSES:
    winner = board.check_win()
    if winner is not None:
        fitness[winner] += 200
    else: # small bonus for drawing
        fitness[0] += 10
    
    return fitness[0]