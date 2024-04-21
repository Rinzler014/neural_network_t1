import random
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate
        self.errors = []

    #Función en la que el modelo predice un resultado de acuerdo a lo que recibe y al valor de los pesos
    def predict(self, inputs):
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return 1 if net_input >= 0 else 0

    #Función que actualiza los pesos
    def update_weights(self, inputs, target):
        #El modelo predice un resultado con la función predict y calcula la diferencia entre
        #el objetivo y la predicción del modelo para asignarlo al error
        prediction = self.predict(inputs)
        error = target - prediction
        #En este ciclo for de acuerdo a la longitud de nuestra matriz de errores vamos
        #actualizando los errores con respecto a nuestra constante de aprendizaje
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += self.learning_rate * error
            else:
                self.weights[i] += self.learning_rate * error * inputs[i-1]

    #Función encargada del entrenamiento del modelo
    def train(self, training_inputs, targets, epochs):
        #Ciclo for para las épocas previamente definidas
        for epoch in range(epochs):
            #Iniciamos en 0 nuestro error
            epoch_errors = 0
            #Ciclo for en el que vamos a actualizar nuestros pesos y errores en las épocas en caso
            #de que el modelo fallé al predecir un resultado
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