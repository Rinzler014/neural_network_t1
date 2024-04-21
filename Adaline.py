import random
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

class Adaline:
    # Inicialización de la clase Adaline definiendo matriz de peso, constante de aprendizaje, matriz de costos y errores
    def __init__(self, input_size, learning_rate=0.1):
        # Inicialización de la clase Adaline con pesos aleatorios
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate
        self.errors = []

    # Función de predicción que cálcula la suma del peso de las entradas
    def predict(self, inputs):
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return self.sigmoid(net_input)  #Aplicamos la función sigmoide

    # Función de activación del sigmoide
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Función para actualizar pesos de acuerdo al error calculado
    def update_weights(self, inputs, target):
        # Calcular la salida predicha y el error
        predicted = self.predict(inputs)
        error = target - predicted

        # Actualizarmos pesos
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += self.learning_rate * error
            else:
                self.weights[i] += self.learning_rate * error * inputs[i-1]

    # Función para calcular el error promedio
    def calculate_errors(self, inputs, targets):
        predictions = [self.predict(inputs[i]) for i in range(len(inputs))]
        errors = [(predictions[i] - targets[i]) ** 2 for i in range(len(targets))]
        return np.mean(errors)
    
    # Función encargada del entrenamiento del modelo
    def train(self, training_inputs, targets, epochs):
        # Ciclo for sobre la cantidad de épocas
        for epoch in range(epochs):
            # Itera sobre cada dato de entrenamiento y objetivos
            for inputs, target in zip(training_inputs, targets):
                # Actualiza pesos
                self.update_weights(inputs, target)
            # Cálcula y guarda el error de cada época
            error = self.calculate_errors(training_inputs, targets)
            self.errors.append(error)

    # Función para probar el modelo con los datos de entrada y calcular la precisión
    def test(self, test_inputs, targets):
        correct = 0 # Inicializamos el contador de predicciones correctas
        total = len(test_inputs)

        print("Pruebas:")
        for inputs, target in zip(test_inputs, targets):
            # Predicción con el modelo
            prediction = self.predict(inputs)
            # Convertimos la salida en binario
            if prediction >= 0.5:
                output = 1
            else:
                output = 0
            print(f"Entrada: {inputs}, Objetivo: {target}, Predicho: {output}")

            if output == target:
                # Si es correcta la salida entonces incrementamos el contador de correctos
                correct += 1

        # Calcular el error y la precisión en una escala del 0 al 1
        error = self.calculate_errors(test_inputs, targets) / len(test_inputs)  # Mean error
        accuracy = correct / total  # Accuracy

        # Imprimimos el error promedio y la precisión
        print(f"Error promedio: {error:.2f}")
        print(f"Precisión: {accuracy:.2f}")

    def test_continous(self, test_inputs, test_targets):
            
        predictions = [self.predict(inputs) for inputs in test_inputs]
        
        # Calcular el error medio
        errors = [(prediction - target) ** 2 for prediction, target in zip(predictions, test_targets)]
        mean_error = np.mean(errors)
        
        # Calcular la precisión
        total_samples = len(test_targets)
        within_tolerance = sum(1 for prediction, target in zip(predictions, test_targets) if abs(prediction - target) < 0.5)
        accuracy = within_tolerance / total_samples
        
        print(f"Error promedio: {mean_error:.2f}")
        print(f"Precisión: {accuracy:.2f}")

    def plot_errors(self):
        # Función para imprimir costos durante el entrenamiento
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Epocas')
        plt.ylabel('Error')
        plt.title('Error por epoca')
        plt.grid(True)
        plt.show()

    def plot_predictions(self, test_inputs, targets):
        # Función para modelar predicciones con los valores esperados
        predictions = [self.predict(inputs) for inputs in test_inputs]
        plt.plot(range(len(predictions)), predictions, label='Predecido')
        plt.plot(range(len(targets)), targets, label='Valor Real')
        plt.title('Predecido contra el valor real')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions_line(self, test_inputs, test_targets):
        
        plt.plot(test_targets, [self.predict(inputs) for inputs in test_inputs], color='blue')
        plt.plot(test_targets, test_targets, color='red')  # Línea diagonal perfecta
        plt.xlabel('Objetivo')
        plt.ylabel('Predicción')
        plt.title('Predicccion contra valor real')
        plt.grid(True)
        plt.show()
    

def case_one():

    adaline_model = Adaline(input_size=4, learning_rate=0.05)
    
    inputs = [ [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1],[1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1] ]

    desired_outputs = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]

    adaline_model.train(inputs, desired_outputs, epochs=100)
    adaline_model.test(inputs, desired_outputs)
    adaline_model.plot_errors()

def case_two():
    
    constant = 4
    training_data = []
    test_data = []
    
    training_patterns = []
    training_outputs = []
    
    test_patterns = []
    test_outputs = []
    
    xt = random.random()
    
    for i in range(1127): 
        
        eq = constant * xt * (1 - xt)
        
        xt = eq
        
        if i < 19:
            pass
        
        if i >= 19 and i < 1023:
            training_data.append(eq)

        if i >= 1023:
            test_data.append(eq)
            
    
    for i in range(len(training_data)):
            if i == 999:
                x1 = training_data[i]
                x2 = training_data[i+1]
                x3 = training_data[i+2]
                result = training_data[i+3]
                
                # Calcular el cuadrado de cada entrada
                x1_squared = x1 ** 2
                x2_squared = x2 ** 2
                x3_squared = x3 ** 2
                
                # Agregar las entradas originales al arreglo de entrenamiento
                training_patterns.append([x1, x2, x3, x1_squared, x2_squared, x3_squared])
                training_outputs.append(result)
                
                break
    
            x1 = training_data[i]
            x2 = training_data[i+1]
            x3 = training_data[i+2]
            result = training_data[i+3]
            
            # Calcular el cuadrado de cada entrada
            x1_squared = x1 ** 2
            x2_squared = x2 ** 2
            x3_squared = x3 ** 2
            
            # Agregar las entradas originales al arreglo de entrenamiento
            training_patterns.append([x1, x2, x3, x1_squared, x2_squared, x3_squared])
            training_outputs.append(result)
    
    for i in range(len(test_data)):
        
        if i == 99:
        
            x1 = test_data[i]
            x2 = test_data[i+1]
            x3 = test_data[i+2]
            result = test_data[i+3]
            
            # Calcular el cuadrado de cada entrada
            x1_squared = x1 ** 2
            x2_squared = x2 ** 2
            x3_squared = x3 ** 2
            
            # Agregar las entradas originales al arreglo de entrenamiento
            test_patterns.append([x1, x2, x3, x1_squared, x2_squared, x3_squared])
            test_outputs.append(result)

            break

        x1 = test_data[i]
        x2 = test_data[i+1]
        x3 = test_data[i+2]
        result = test_data[i+3]
        
        # Calcular el cuadrado de cada entrada
        x1_squared = x1 ** 2
        x2_squared = x2 ** 2
        x3_squared = x3 ** 2
        
        # Agregar las entradas originales al arreglo de entrenamiento
        test_patterns.append([x1, x2, x3, x1_squared, x2_squared, x3_squared])
        test_outputs.append(result)
        
    
    adaline_model = Adaline(input_size=6, learning_rate=0.05)
    adaline_model.train(training_patterns, training_outputs, epochs=1000)
    adaline_model.test_continous(test_patterns, test_outputs)
    adaline_model.plot_errors()
    adaline_model.plot_predictions(test_patterns, test_outputs)
    adaline_model.plot_predictions_line(test_patterns, test_outputs)

os.system('cls')
case_two()
