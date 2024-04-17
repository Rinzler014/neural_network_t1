

class Adaline:
    
    """
    Class for the Adaline model.
    
    Parameters:
    
    input_size: int
        The number of input features.
    learning_rate: float
        The learning rate for the model.
        
        
    Methods:
    
    predict(inputs): float
        Predicts the output for the given inputs.
    update_weights(inputs, target): None
        Updates the weights and bias based on the error.
    train(training_inputs, targets, epochs): None
        Trains the model for a specific number of epochs.
    test(test_inputs, targets): None
        Tests the model with test data and calculates the accuracy.        
    """
    
    def __init__(self, input_size, learning_rate=0.1):
        # Inicialización de la clase Adaline
        self.weights = [6] * input_size  # Inicializamos los pesos a cero
        self.bias = 0.25  # Inicializamos el sesgo a cero
        self.learning_rate = learning_rate  # Tasa de aprendizaje

    def predict(self, inputs):
        # Función de predicción que calcula la suma ponderada de las entradas
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights, inputs)) + self.bias
        return net_input

    def update_weights(self, inputs, target):
        # Función para actualizar los pesos y el sesgo basado en el error
        predicted = self.predict(inputs)  # Calculamos la salida predicha
        error = target - predicted  # Calculamos el error
        
        print(f"Predicted: {predicted}, Target: {target}, Error: {error}")

        # Actualizamos los pesos
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]

        # Actualizamos el sesgo
        self.bias += self.learning_rate * error

    def train(self, training_inputs, targets, epochs):
        # Función para entrenar el modelo durante un número específico de épocas
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}:")
            # Iteramos sobre cada par de datos de entrenamiento y objetivo
            for inputs, target in zip(training_inputs, targets):
                # Actualizamos los pesos y el sesgo
                self.update_weights(inputs, target)
            print()

    def test(self, test_inputs, targets):
            correct = 0
            total = len(test_inputs)
    
            print("Testing:")
            for inputs, target in zip(test_inputs, targets):
                prediction = self.predict(inputs)
                if prediction >= 0.0:
                    output = 1
                else:
                    output = 0
                print(f"Input: {inputs}, Target: {target}, Predicted: {output}, Prediction: {prediction}")
                
                # Comparamos la predicción con el valor real
                if output == target:
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

adaline_model = Adaline(input_size=4, learning_rate=0.25)
adaline_model.train(inputs, desired_outputs, epochs=2)

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