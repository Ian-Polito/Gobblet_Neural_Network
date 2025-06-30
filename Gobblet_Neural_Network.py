import argparse
import numpy as np
from gobblet.board import Board

parser = argparse.ArgumentParser()
parser.add_argument("-visualization",action="store_true",help="Enable visualization output.")
args = parser.parse_args()

board = Board()

#start visualization if flag is set
if args.visualization:
    board.initialize_visualization()
    visualization = True

board.game_loop()
board.visualize_board_wrapper()