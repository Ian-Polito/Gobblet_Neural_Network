import argparse
import numpy as np
from gobblet.board import Board

parser = argparse.ArgumentParser()
parser.add_argument("-visualization",action="store_true",help="Enable visualization output.")
args = parser.parse_args()

board = Board()
game_state = "ongoing"

#start visualization if flag is set
if args.visualization:
    board.initialize_visualization()
    visualization = True

board.game_loop()
board.visualize_board_wrapper()
#while game_state == "ongoing":
    #board.visualize_board_wrapper()
    #before_move = [row[:] for row in board.grid]
    # simulate a game by making random moves for now
    # decide move
    # check if its valid
    # if an opponent's piece is underneath, do an uncover check
    # if returned true opponent wins
    # move piece
    # check win
    # if None, continue game
    