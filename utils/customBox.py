import numpy as np

class CustomBox:
    def __init__(self, low, high, dtype=np.float32):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        
    @property
    def shape(self):
        # Both self.low and self.high should have the same shape,
        # so returning either one is valid.
        return self.low.shape
