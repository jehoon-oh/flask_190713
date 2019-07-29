import tensorflow as tf

class CalculatorModel:
    def __init__(self):
        self.a = 0
        self.b = 0

    def input_number(self):
        self.a = int(input('1st number\n'))
        self.b = int(input('2nd number\n'))

    def hook(self, flag):
        self.input_number()
        if flag == 1: result = self.plus()
        elif flag == 2: result = self.subtract()
        elif flag == 3: result = self.multiply()
        elif flag == 4: result = self.divide()
        return tf.keras.backend.eval(result)

    @tf.function
    def plus(self):
        result = tf.add(self.a, self.b)
        return result

    @tf.function
    def subtract(self):
        result = tf.subtract(self.a, self.b)
        return result

    @tf.function
    def multiply(self):
        result = tf.multiply(self.a, self.b)
        return result

    @tf.function
    def divide(self):
        result = tf.divide(self.a, self.b)
        return result