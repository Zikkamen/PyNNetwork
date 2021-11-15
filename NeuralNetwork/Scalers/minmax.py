import numpy as np

class scalers():

    def __init__(self, Matrix_orig):
        self.Mat_orig = Matrix_orig
        self.Mat_scal = Matrix_orig
    
    def getOrigMatrix(self):
        return self.Mat_orig 

    def min_max_scal(self):
        X     = np.array(X)
        x_min = min(X)
        x_max = max(X)

        if x_max - x_min == 0: return [0 for i in X]
        
        X = [(x - x_min)/(x_max - x_min) for x in X]
        return X

    def min_max_rescal(self, X_original):
        X     = np.array(X)
        x_min = min(self.Mat_orig)
        x_max = max(self.Mat_orig)

        X = [x*(x_max - x_min) + x_min for x in X]

        return np.array(X)