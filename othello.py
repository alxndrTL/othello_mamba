
"""

adapted, simplified from https://github.com/likenneth/othello_world/blob/master/data/othello.py

"""

import numpy as np
import random

rows = list("abcdefgh")
columns = [str(i) for i in range(1, 9)]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

def move_to_str(move):
    r, c = move // 8, move % 8
    return "".join([rows[r], columns[c]])

def generate_game(_):
    tbr = []
    game = OthelloGame()

    possible_next_steps = game.get_valid_moves()
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        game.update([next_step,])
        possible_next_steps = game.get_valid_moves()

    return tbr

class OthelloGame():
    def __init__(self, board_size = 8):
        self.board_size = board_size * board_size

        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state

        self.next_hand_color = 1 # 1 is black, -1 is white
        self.history = []

    @staticmethod
    def get_tbf(state, color, move):
        """
        given a game state, color and a move (`color` plays `move`),
        returns tbf, a list containing all 1-color pieces to be flipped
        """

        r, c = move // 8, move % 8
        tbf = []

        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if state[cur_r, cur_c] == 0:
                    break
                elif state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        return tbf
    
    def update(self, moves, prt=False):
        """
        takes a new move or new moves and update state
        """

        if prt:
            self.__print__()
        for move in moves:
            self.play_move(move)
            if prt:
                self.__print__()

    def play_move(self, move):
        """
        play a move
        """

        r, c = move // 8, move % 8

        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"

        # get all pieces to be flipped (tbf)
        tbf = self.get_tbf(self.state, self.next_hand_color, move)

        # means current hand is forfeited : we switch
        if len(tbf) == 0:  
            self.next_hand_color *= -1
            tbf = self.get_tbf(self.state, self.next_hand_color, move)

        # either move was illegal, either the game must have ended
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
        
        # play the move and flip the pieces to be flipped
        self.state[r, c] = self.next_hand_color
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1

        # hand is switched for next turn
        self.next_hand_color *= -1
        self.history.append(move)
        
    def tentative_move(self, move):
        """
        tentatively put a piece, do nothing to state

        returns 0 if this is not a move at all: occupied or both player have to forfeit (game ended)
        return 1 if regular move
        return 2 if forfeit happens but the opponent can drop piece at this place
        """

        r, c = move // 8, move % 8

        if not self.state[r, c] == 0:
            return 0

        color = self.next_hand_color
        tbf = self.get_tbf(self.state, color, move)

        if len(tbf) != 0:
            return 1
        
        # means current hand is forfeited : we switch
        else:
            color *= -1
            tbf = self.get_tbf(self.state, self.next_hand_color, move)

            if len(tbf) == 0:
                return 0
            else:
                return 2
        
    def get_valid_moves(self):
        """
        Returns all the valid (legal) moves given the current state of the game
        """
        
        regular_moves = []
        forfeit_moves = []

        for move in range(64):
            x = self.tentative_move(move)
            # x = 0 : not a valid move OR both player have to forfeit
            # x = 1 : valid move
            # x = 2 : valid move, but for the other player (ie, the current player forfeits)

            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)

        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
    
    def __print__(self):
        """
        Prints the current state of the game, as well as all the moves.
        """

        print("-" * 20)

        print([move_to_str(move) for move in self.history])

        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            row_to_print = []
            for el in row:
                if el == -1:
                    row_to_print.append("O") # white
                elif el == 0:
                    row_to_print.append(" ") # empty
                else:
                    row_to_print.append("X") # black

            print(" ".join([a[k]] + row_to_print))

        row_to_print = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + row_to_print))

        print("-" * 20)
