class TransferFunction:
    def __call__(self, x: float) -> float:
        raise NotImplementedError

class Heaviside(TransferFunction):
    """Translatable Heaviside Step Function"""
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, x: float) -> float:
        if x >= self.threshold:
            return 1.0
        else:
            return 0.0
