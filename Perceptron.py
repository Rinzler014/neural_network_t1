import random
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate
        self.errors = []

    def predict(self, inputs):
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return 1 if net_input >= 0 else 0

    def update_weights(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += self.learning_rate * error
            else:
                self.weights[i] += self.learning_rate * error * inputs[i-1]

    def train(self, training_inputs, targets, epochs):
        for epoch in range(epochs):
            epoch_errors = 0
            for inputs, target in zip(training_inputs, targets):
                self.update_weights(inputs, target)
                epoch_errors += int(target != self.predict(inputs))
            self.errors.append(epoch_errors)

    def test(self, test_inputs, targets):
        correct = 0
        total = len(test_inputs)
        print("Testing:")
        for inputs, target in zip(test_inputs, targets):
            prediction = self.predict(inputs)
            print(f"Input: {inputs}, Target: {target}, Predicted: {prediction}")
            if prediction == target:
                correct += 1
                
        mean_training_error = sum(self.errors) / len(self.errors)
        accuracy = correct / total
        print(f"Mean error: {mean_training_error:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

    def plot_errors(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.title('Errors vs. Epochs')
        plt.grid(True)
        plt.show()


def case_one():

    # Datos de entrenamiento y prueba
    inputs = [ [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1],[1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1] ]

    desired_outputs = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]

    # Crear un perceptrón y entrenarlo
    perceptron = Perceptron(input_size=4, learning_rate=0.1)
    perceptron.train(inputs, desired_outputs, epochs=100)

    # Probar el perceptrón entrenado
    perceptron.test(inputs, desired_outputs)

    # Graficar los errores durante el entrenamiento
    perceptron.plot_errors()
    
case_one()