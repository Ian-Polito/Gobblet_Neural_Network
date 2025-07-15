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
    
def play_game(net1, net2, max_turns=32):
    board = Board()
    players = [net1, net2]
    fitness = [0, 0]
    turn = 0
    
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
                    return 0 if board.current_player == 0 else 1
                else:
                    board.move_piece(None, from_row, from_col, to_row, to_col)
                    fitness[board.current_player] += 5
                    valid = True
        if not valid:
            fitness[board.current_player] -= 1
            board.current_player = 1 - board.current_player # end turn
        turn += 1
    winner = board.check_win()
    if winner is not None:
        fitness[winner] += 100
    #else: # small bonus for drawing, commented out for now until population matures
        #fitness[0] += 10
    return fitness[0]