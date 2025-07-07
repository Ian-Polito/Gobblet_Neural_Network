import neat
import numpy as np
from .board import Board

def evaluate_genome(genome, config, opponents):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    
    for opponent_genome in opponents:
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
        result = play_game(net, opponent_net)
        if result == 1:
            fitness += 100 # win
        elif result == 0:
            fitness += 0 # loss
        else:
            fitness += 10 # draw
    genome.fitness = fitness / len(opponents)
    
def play_game(net1, net2, max_turns=300):
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
        return 1 if winner == 0 else 0
    return "draw"

def eval_genomes(genomes, config):
    for i, (genome_id, genome) in enumerate(genomes):
        opponents = [g for j, (gid, g) in enumerate(genomes) if j != i]
        evaluate_genome(genome, config, opponents[:3])