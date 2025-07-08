import neat
import numpy as np
import multiprocessing
import os
from .board import Board

def evaluate_genome(genome, config, opponents):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    
    for opponent_genome in opponents:
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
        fitness += play_game(net, opponent_net)
    return fitness / len(opponents)
    
def evaluate_genome_wrapper(args):
    genome, config, all_genomes, index=args
    opponents = [g for j, (gid, g) in enumerate(all_genomes) if j != index][:3]
    return evaluate_genome(genome, config, opponents)
    
def eval_genomes(genomes, config):
    num_cores = max(1, os.cpu_count() - 2) # leave 2 cores free
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(evaluate_genome_wrapper, [(genome, config, genomes, i) for i, (gid, genome) in enumerate(genomes)])
        for (gid, genome), fitness in zip(genomes, results):
            genome.fitness = fitness
    
def play_game(net1, net2, max_turns=100):
    board = Board()
    players = [net1, net2]
    fitness = [0, 0]
    turn = 0
    
    while board.check_win() is None and turn < max_turns:
        current_net = players[board.current_player]
        inputs = board.encode_board()
        outputs = current_net.activate(inputs)
        move_index = np.argmax(outputs)
        move = board.decode_move(move_index)
        
        valid = False
        if move[0] == "external":
            stack_index, to_row, to_col = move[1:]
            if board.is_valid_move(board.current_player, stack_index, to_row, to_col):
                board.move_piece(stack_index, None, None, to_row, to_col)
                fitness[board.current_player] += 1
                valid = True
        elif move[0] == "board":
            from_row, from_col, to_row, to_col = move[1:]
            from_stack = board.grid[from_row][from_col]
            if board.is_valid_move(board.current_player, from_stack, to_row, to_col):
                if board.uncover_check(from_row, from_col, to_row, to_col):
                    # move results in a loss due to uncovering a win without interrupting it
                    return 0 if board.current_player == 0 else 1
                else:
                    board.move_piece(None, from_row, from_col, to_row, to_col)
                    fitness[board.current_player] += 1
                    valid = True
        if not valid:
            fitness[board.current_player] -= 5
            board.current_player = 1 - board.current_player # end turn
        turn += 1
    winner = board.check_win()
    if winner is not None:
        fitness[winner] += 100
    #else: # small bonus for drawing, commented out for now until population matures
        #fitness[0] += 10
        #fitness[1] += 10
    return fitness[0]