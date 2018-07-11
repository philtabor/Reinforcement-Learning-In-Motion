def printV(V, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):            
            state = grid.m * idx + idy 
            print('%.2f' % V[state], end='\t')
        print('\n')
    print('--------------------')