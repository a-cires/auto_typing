
class MotorMotionModel:
    def __init__(self):
        # Parameters derived from data (in mm/s)
        self.models = {
            'left': {
                'slope': 2.523876556578235,
                'intercept': 12.749577693557118
            },
            'right': {
                'slope': -0.9399657947686118,
                'intercept': 35.807557344064385
            }
        }

    def estimate_position(self, direction, time_sec):
        model = self.models[direction]
        return model['slope'] * time_sec + model['intercept']

    def estimate_time_to_travel(self, direction, distance_mm):
        model = self.models[direction]
        slope = model['slope']
        intercept = model['intercept']
        if slope == 0:
            raise ValueError("Slope is zero; cannot compute time.")
        return (distance_mm - intercept) / slope
