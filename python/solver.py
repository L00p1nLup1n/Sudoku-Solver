"""Backtracking Sudoku solver (ported from Java implementation)."""
SIZE = 9

def is_valid(board, row, col, num):
    for c in range(SIZE):
        if board[row][c] == num:
            return False
    for r in range(SIZE):
        if board[r][col] == num:
            return False
    box_row = row - row % 3
    box_col = col - col % 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False
    return True


def solve(board):
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row][col] == 0:
                for num in range(1, SIZE + 1):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def is_board_valid(board):
    size = len(board)
    clue_count = 0
    for i in range(size):
        row_check = [False] * (size + 1)
        col_check = [False] * (size + 1)
        for j in range(size):
            row_val = board[i][j]
            col_val = board[j][i]
            if row_val != 0:
                if row_check[row_val]:
                    return False
                row_check[row_val] = True
                clue_count += 1
            if col_val != 0:
                if col_check[col_val]:
                    return False
                col_check[col_val] = True
                clue_count += 1
    for box_row in range(3):
        for box_col in range(3):
            box_check = [False] * (size + 1)
            for r in range(box_row * 3, box_row * 3 + 3):
                for c in range(box_col * 3, box_col * 3 + 3):
                    val = board[r][c]
                    if val != 0:
                        if box_check[val]:
                            return False
                        box_check[val] = True
                        clue_count += 1
    return clue_count >= 17
