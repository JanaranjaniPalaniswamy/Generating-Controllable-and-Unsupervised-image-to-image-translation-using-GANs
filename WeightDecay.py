import tensorflow as tf
import tensorflow.keras as keras

class WeightDecay:
    
    def __init__(self, initial_weight_value, total_step, step_decay):
        self._initial_weight_value = initial_weight_value
        self._steps = total_step
        self._step_decay = step_decay
        self.current_weight_value = tf.Variable(initial_value=initial_weight_value, trainable=False, dtype=tf.float32)
        
    def currentValue(self, step):
        if step >= self._step_decay:
            self.current_weight_value.assign(self._initial_weight_value * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)))
        else:
            self.current_weight_value.assign(self._initial_weight_value)
        return self.current_weight_value