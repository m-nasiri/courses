#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:36:09 2017

@author: majid_nasiri
"""

import numpy as np
import itertools

N = 6                           # Number of Queens
itr = 0                         # Counter for produced combination
best_combination = []           # list of correct combination


# cost function for evaluationg how well the input combination is.
# low value refer to less conflict between queens therefore boad 
# with zero cost is a solution for problem.
def promissing(board):
    board_len = len(board)
    conflict = 0
    for i in range(board_len):
        for j in range(i, board_len):
            if (i != j):
                # check for horizontal threats
                #                             check for diagonal threats
                if ((board[i] == board[j]) or np.abs(board[i]-board[j]) == (j-i)):
                    conflict += 1
    return conflict


# produce all possible combination using code in
# https://gist.github.com/3997853
for combination in itertools.product(range(N), repeat=N):
    itr += 1
    comb = np.asarray(combination, dtype=np.int8)
    cost = promissing(comb)
    print('combination',itr,'=',comb, 'cost = ', cost)
    if (cost == 0):
        best_combination.append(comb)       # save solution combination


# visualization chess board
board1 = ('- '*N)
for i in best_combination:
    print('Board '+str(i))
    for j in range(N):
        board = list(board1)
        board[2*i[j]] = '@'
        board = ''.join(board)
        print(board)
    







    
        