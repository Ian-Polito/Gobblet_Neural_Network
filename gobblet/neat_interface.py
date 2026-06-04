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
    
def play_game(net1, net2, max_turns=24):
    board = Board()
    players = [net1, net2]
    fitness = [0, 0]
    turn = 0
    invalid_streak = [0, 0]
    
    # Track exploration metrics for both players
    stacks_used = [set(), set()]  # Track which external stacks each player has used
    board_positions_used = [set(), set()]  # Track board positions each player has used
    recent_moves = [[], []]  # Track recent moves to penalize repetition
    repetition_count = [0, 0]  # Track consecutive repetitions for escalating penalties
    
    # temporarily restrict outputs to just external stack -> board moves
    allowed_indices = list(range(48))
    
    # Add adjacent board→board moves only
    for move_idx in range(48, 288):
        move = board.decode_move(move_idx)
        if move[0] == "board":
            from_row, from_col, to_row, to_col = move[1:]
            # Only allow moves to adjacent squares (Manhattan distance of 1)
            if abs(from_row - to_row) + abs(from_col - to_col) == 1:
                allowed_indices.append(move_idx)
    
    while board.check_win() is None and turn < max_turns:
        # check if board is full (all 16 positions occupied) - early draw condition
        # should be removed once temporary masking is removed and gobbling can occur
        board_full = all(board.grid[row][col] for row in range(4) for col in range(4))
        if board_full:
            break # end game early, no more pieces can be placed
            
        current_net = players[board.current_player]
        inputs = board.encode_board()
        outputs = current_net.activate(inputs)
        # temporarily mask outputs
        masked_outputs = [outputs[i] if i in allowed_indices else -float("inf") for i in range(len(outputs))]
        move_index = np.argmax(masked_outputs)
        #move_index = np.argmax(outputs)
        move = board.decode_move(move_index)
        
        # repetition penalty
        if move in recent_moves[board.current_player][-2:]:
            repetition_count[board.current_player] += 1
            penalty = 10 + (repetition_count[board.current_player] * 10)
            fitness[board.current_player] -= penalty
        else:
            repetition_count[board.current_player] = 0  # Reset if they break the pattern
        
        valid = False
        if move[0] == "external":
            stack_index, to_row, to_col = move[1:]
            # partial reward
            if board.external_stacks[board.current_player][stack_index]:
                piece = board.external_stacks[board.current_player][stack_index][-1]
                if not board.grid[to_row][to_col] or board.grid[to_row][to_col][-1].size < piece.size:
                    fitness[board.current_player] += 2
            if board.is_valid_move(board.current_player, stack_index, to_row, to_col):
                board.move_piece(stack_index, None, None, to_row, to_col)
                fitness[board.current_player] += 5
                
                # EXPLORATION BONUSES:
                # Bonus for using a new external stack
                if stack_index not in stacks_used[board.current_player]:
                    stacks_used[board.current_player].add(stack_index)
                    fitness[board.current_player] += 15  # Increased new stack bonus
                
                # Bonus for using a new board position, commented out for now
                #position = (to_row, to_col)
                #if position not in board_positions_used[board.current_player]:
                    #board_positions_used[board.current_player].add(position)
                    #fitness[board.current_player] += 3  # New position bonus
                
                valid = True
                
        elif move[0] == "board":
            from_row, from_col, to_row, to_col = move[1:]
            from_stack = board.grid[from_row][from_col]
            # partial reward
            if from_stack:
                piece = from_stack[-1]
                if piece.owner == board.current_player:
                    if not board.grid[to_row][to_col] or board.grid[to_row][to_col][-1].size < piece.size:
                        fitness[board.current_player] += 2
            if board.is_valid_move(board.current_player, from_stack, to_row, to_col):
                if board.uncover_check(from_row, from_col, to_row, to_col):
                    # move results in a loss due to uncovering a win without interrupting it
                    fitness[1 - board.current_player] += 100
                    return fitness[0]
                else:
                    board.move_piece(None, from_row, from_col, to_row, to_col)
                    fitness[board.current_player] += 5
                    # Bonus for successfully using a board->board move
                    fitness[board.current_player] += 8  # Encourage using board->board moves
                    valid = True
        
        # STRATEGIC BONUSES: Check for pieces in a line after placing
        if valid:
            line_bonus = calculate_line_bonus(board, board.current_player, to_row, to_col)
            fitness[board.current_player] += line_bonus
        
        # Track recent moves for repetition checking
        recent_moves[board.current_player].append(move)
        if len(recent_moves[board.current_player]) > 3:  # Keep only last 3 moves
            recent_moves[board.current_player].pop(0)
            
        if not valid:
            invalid_streak[board.current_player] += 1
            fitness[board.current_player] -= 2 * invalid_streak[board.current_player]
            board.current_player = 1 - board.current_player # end turn
        else:
            invalid_streak[board.current_player] = 0 # reset streak on valid move
        turn += 1
    
    # END OF GAME BONUSES:
    winner = board.check_win()
    if winner is not None:
        fitness[winner] += 100
    #else: # small bonus for drawing, commented out for now until population matures
        #fitness[0] += 10
    
    # Exploration bonuses at end of game
    for player in [0, 1]:
        # Strong bonus for stack diversity (reward using multiple stacks)
        stack_diversity_bonus = len(stacks_used[player]) * 12
        if len(stacks_used[player]) >= 2:  # Extra bonus for using multiple stacks
            stack_diversity_bonus += 20
        fitness[player] += stack_diversity_bonus
        
        # Bonus for board coverage (reward spreading pieces around)
        fitness[player] += len(board_positions_used[player]) * 2
        
        # Efficiency bonus (more pieces placed = better)
        pieces_placed = sum(1 for row in board.grid for stack in row if stack and stack[-1].owner == player)
        fitness[player] += pieces_placed * 3
    
    return fitness[0]