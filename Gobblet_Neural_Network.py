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

if __name__ == "__main__":
    config_path = "config-feedforward.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path)
    
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
    p.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix="Gobblet_Population-"))
    
    # run NEAT
    winner = p.run(eval_genomes, n=args.generations)
    final_population = [(k, v) for k, v in p.population.items()]
    top_two = sorted(final_population, key=lambda g: g[1].fitness if g[1].fitness is not None else -float('inf'), reverse=True)[:2]
    
    # Manually save a checkpoint after the run
    p.checkpointer.save_checkpoint(p, p.generation)
    
    # Save the top two genomes
    with open("gobblet_champion1.pk1", "wb") as f:
        pickle.dump(top_two[0][1], f)
    if len(top_two) > 1:
        with open("gobblet_champion2.pk1", "wb") as f:
            pickle.dump(top_two[1][1], f)
            
    print("Training complete. Best genome saved to gobblet_champion.pk1")
    
    #start visualization if flag is set
    if args.visualization:
        board = Board()
        board.initialize_visualization()
        visualization = True
        board.game_loop()
        board.visualize_board_wrapper()