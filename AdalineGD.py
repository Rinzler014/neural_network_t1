import numpy as np
import random

class Adaline:
    def __init__(self, input_size, learning_rate=0.1):
        # Inicialización de la clase Adaline con pesos aleatorios
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]  # Inicialización de pesos aleatorios, incluido w0
        self.learning_rate = learning_rate  # Tasa de aprendizaje

    def predict(self, inputs):
        # Función de predicción que calcula la suma ponderada de las entradas
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return net_input

    def update_weights(self, inputs, target):
        # Función para actualizar los pesos basado en el error
        predicted = self.predict(inputs)  # Calculamos la salida predicha
        error = target - predicted  # Calculamos el error

        # Actualizamos los pesos
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += self.learning_rate * error  # Actualizamos w0 (sesgo)
            else:
                self.weights[i] += self.learning_rate * error * inputs[i-1]  # Actualizamos los otros pesos

    def train(self, training_inputs, targets, epochs):
        # Función para entrenar el modelo durante un número específico de épocas
        for epoch in range(epochs):
            # Iteramos sobre cada par de datos de entrenamiento y objetivo
            for inputs, target in zip(training_inputs, targets):
                # Actualizamos los pesos
                self.update_weights(inputs, target)

    def test(self, test_inputs, targets):
        # Función para probar el modelo con datos de prueba y calcular la precisión
        correct = 0  # Inicializamos el contador de predicciones correctas
        total = len(test_inputs)  # Calculamos el número total de muestras de prueba

        print("Testing:")
        for inputs, target in zip(test_inputs, targets):
            # Realizamos una predicción con el modelo
            prediction = self.predict(inputs)
            # Convertimos la salida a una salida binaria (0 o 1)
            if prediction >= 0.5:
                output = 1
            else:
                output = 0
            print(f"Input: {inputs}, Target: {target}, Predicted: {output}, Prediction: {prediction}")

            if output == target:
                # Si la predicción es correcta, incrementamos el contador
                correct += 1

        # Calculamos la precisión del modelo
        accuracy = (correct / total) * 100.0
        print(f"Accuracy: {accuracy:.2f}%")

inputs = [
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]

desired_outputs = [
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1
]

adaline_model = Adaline(input_size=4, learning_rate=0.1)
adaline_model.train(inputs, desired_outputs, epochs=1000)

test_inputs = [
        [0,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1]
]

test_targets = [
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1
]

adaline_model.test(test_inputs, test_targets)
