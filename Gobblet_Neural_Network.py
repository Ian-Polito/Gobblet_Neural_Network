import argparse
import neat
import os
import pickle
from gobblet.neat_interface import eval_genomes
from gobblet.board import Board

parser = argparse.ArgumentParser()
parser.add_argument("-visualization", action="store_true", help="Enable visualization output for a game.")
parser.add_argument("-generations", type=int, default=50, help="Number of generations to train.")
parser.add_argument("-checkpoint", type=str, help="Path to a checkpoint file to resume training")
args = parser.parse_args()

class HallOfFameReporter(neat.reporting.BaseReporter):
    def __init__(self, hall_of_fame, filename="gobblet_hall_of_fame.pkl", interval=10):
        self.hall_of_fame = hall_of_fame
        self.filename = filename
        self.interval = interval
        self.generation = 0
    
    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        if self.generation % self.interval == 0:
            with open(self.filename, "wb") as f:
                pickle.dump(self.hall_of_fame, f)
            print(f"Hall of Fame saved ({len(self.hall_of_fame)} members)")

def load_hall_of_fame(filename="gobblet_hall_of_fame.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            print(f"Loading Hall of Fame from {filename}")
            return pickle.load(f)
    print("No Hall of Fame found, starting fresh.")
    return []

def make_eval_genomes(hall_of_fame):
    def eval_genomes_wrapper(genomes, config):
        eval_genomes(genomes, config, hall_of_fame)
    return eval_genomes_wrapper

def load_genome(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None

if __name__ == "__main__":
    config_path = "config-feedforward.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path)
    
    # start visualization if flag is set
    if args.visualization:
        genome1 = load_genome("gobblet_champion1.pkl")
        genome2 = load_genome("gobblet_champion2.pkl")
        
        if (genome1 != None and genome2 != None):
            net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
            net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
            board = Board()
            board.initialize_visualization()
            board.game_loop(net1, net2)
            board.visualize_board_wrapper()
        else:
            print("Need two gobblet_champion pkl files to visualize a game. One or both are missing.")
        exit()
    
    # create or resume population
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        p = neat.Checkpointer.restore_checkpoint(args.checkpoint)
        
    else:
        print("Starting new population.")
        p = neat.Population(config)

    # add reporters
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    checkpointer = neat.Checkpointer(generation_interval=10, filename_prefix="Gobblet_Population-")
    p.add_reporter(checkpointer
    p.add_reporter(HallOfFameReporter(hall_of_fame))
    
    # run NEAT
    hall_of_fame = load_hall_of_fame()
    winner = p.run(make_eval_genomes, n=args.generations)
    final_population = [(k, v) for k, v in p.population.items()]
    top_two = sorted(final_population, key=lambda g: g[1].fitness if g[1].fitness is not None else -float('inf'), reverse=True)[:2]
    
    # Save the top two genomes
    with open("gobblet_champion1.pkl", "wb") as f:
        pickle.dump(top_two[0][1], f)
    if len(top_two) > 1:
        with open("gobblet_champion2.pkl", "wb") as f:
            pickle.dump(top_two[1][1], f)
            
    # Save the Hall of Fame
    with open("gobblet_hall_of_fame.pkl", "wb") as f:
        pickle.dump(hall_of_fame, f)
            
    print("Training complete. Best genome saved to gobblet_champion.pkl")