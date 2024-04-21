import random
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    # Función para iniciar pesos y definir valores como la constante de aprendizaje y la matriz de errores
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate
        self.errors = []

    # Función para predecir un resultado de acuerdo al input y los pesos
    def predict(self, inputs):
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return 1 if net_input >= 0 else 0

    # Función para actualizar pesos
    def update_weights(self, inputs, target):
        # El modelo llama a la función predict y calcula la diferencia entre
        # el objetivo y la predicción del modelo
        prediction = self.predict(inputs)
        error = target - prediction
         # Ciclo for para actualizar los errores
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += self.learning_rate * error
            else:
                self.weights[i] += self.learning_rate * error * inputs[i-1]

    # Función encargada del entrenamiento del modelo
    def train(self, training_inputs, targets, epochs):
        # Ciclo for sobre la cantidad de épocas
        for epoch in range(epochs):
            epoch_errors = 0
            # Ciclo for para actualizar pesos y errores en las épocas
            for inputs, target in zip(training_inputs, targets):
                self.update_weights(inputs, target)
                epoch_errors += int(target != self.predict(inputs))
            self.errors.append(epoch_errors)

    # Función para probar el modelo con los datos de entrada y calcular precisión
    def test(self, test_inputs, targets):
        correct = 0
        total = len(test_inputs)
        print("Pruebas:")
        # Ciclo for para mostrar todos los datos de entrada, esperados y predichos
        for inputs, target in zip(test_inputs, targets):
            prediction = self.predict(inputs)
            print(f"Entrada: {inputs}, Objetivo: {target}, Predicho: {prediction}")
            if prediction == target:
                correct += 1
                
        mean_training_error = sum(self.errors) / len(self.errors)
        accuracy = correct / total
        print(f"Error promedio: {mean_training_error:.2f}")
        print(f"Precisión: {accuracy:.2f}")

    # Función para imprimir errores
    def plot_errors(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Épocas')
        plt.ylabel('Errores')
        plt.title('Errores vs. Épocas')
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