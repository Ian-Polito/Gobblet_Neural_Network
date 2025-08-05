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
    
def play_game(net1, net2, max_turns=16):
    board = Board()
    players = [net1, net2]
    fitness = [0, 0]
    turn = 0
    
    # Track exploration metrics for both players
    stacks_used = [set(), set()]  # Track which external stacks each player has used
    board_positions_used = [set(), set()]  # Track board positions each player has used
    recent_moves = [[], []]  # Track recent moves to penalize repetition
    
    # temporarily restrict outputs to just external stack -> board moves
    allowed_indices = list(range(48))
    
    while board.check_win() is None and turn < max_turns:
        current_net = players[board.current_player]
        inputs = board.encode_board()
        outputs = current_net.activate(inputs)
        # temporarily mask outputs
        masked_outputs = [outputs[i] if i in allowed_indices else -float("inf") for i in range(len(outputs))]
        move_index = np.argmax(masked_outputs)
        #move_index = np.argmax(outputs)
        move = board.decode_move(move_index)
        
        # Check for repetition penalty (trying same move as last 2 attempts)
        if move in recent_moves[board.current_player][-2:]:
            fitness[board.current_player] -= 5  # Penalty for repetition
        
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
                valid = True
                
                # EXPLORATION BONUSES:
                # Bonus for using a new external stack
                if stack_index not in stacks_used[board.current_player]:
                    stacks_used[board.current_player].add(stack_index)
                    fitness[board.current_player] += 8  # New stack bonus
                
                # Bonus for using a new board position
                position = (to_row, to_col)
                if position not in board_positions_used[board.current_player]:
                    board_positions_used[board.current_player].add(position)
                    fitness[board.current_player] += 3  # New position bonus
                
                # Bonus for board coverage (using different quadrants)
                quadrant = (to_row // 2, to_col // 2)
                player_quadrants = {(pos[0] // 2, pos[1] // 2) for pos in board_positions_used[board.current_player]}
                if len(player_quadrants) > len(player_quadrants) - (1 if quadrant in player_quadrants else 0):
                    fitness[board.current_player] += 4  # New quadrant bonus
                
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
                    valid = True
                    
        # Track recent moves for repetition checking
        recent_moves[board.current_player].append(move)
        if len(recent_moves[board.current_player]) > 3:  # Keep only last 3 moves
            recent_moves[board.current_player].pop(0)
                    
        if not valid:
            fitness[board.current_player] -= 1
            board.current_player = 1 - board.current_player # end turn
        turn += 1
    winner = board.check_win()
    if winner is not None:
        fitness[winner] += 100
    #else: # small bonus for drawing, commented out for now until population matures
        #fitness[0] += 10
        
    # Exploration bonuses at end of game
    for player in [0, 1]:
        # Bonus for stack diversity (reward using multiple stacks)
        fitness[player] += len(stacks_used[player]) * 6
        
        # Bonus for board coverage (reward spreading pieces around)
        fitness[player] += len(board_positions_used[player]) * 2
        
        # Efficiency bonus (more pieces placed = better)
        pieces_placed = sum(1 for row in board.grid for stack in row if stack and stack[-1].owner == player)
        fitness[player] += pieces_placed * 3
        
    return fitness[0]