"""Split a warped square board image into 81 cell images."""
def split_cells_from_array(warped):
    import cv2

    size = warped.shape[0]
    cell_size = size // 9
    margin = cell_size // 10
    cells = []
    for row in range(9):
        for col in range(9):
            x = col * cell_size + margin
            y = row * cell_size + margin
            w = cell_size - 2 * margin
            h = cell_size - 2 * margin
            roi = warped[y:y+h, x:x+w]
            resized = cv2.resize(roi, (28, 28))
            cells.append(resized)
    return cells
