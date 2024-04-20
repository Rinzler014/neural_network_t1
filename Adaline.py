import random
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, input_size, learning_rate=0.1):
        # Adaline class initialization with random weights
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]  # Initialize random weights, including w0
        self.learning_rate = learning_rate  # Learning rate
        self.costs = []  # List to store costs during training

    def predict(self, inputs):
        # Prediction function that calculates the weighted sum of inputs
        net_input = sum(weight * input_value for weight, input_value in zip(self.weights[1:], inputs)) + self.weights[0]
        return self.sigmoid(net_input)  # Apply sigmoid function

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def update_weights(self, inputs, target):
        # Function to update weights based on error
        predicted = self.predict(inputs)  # Calculate predicted output
        error = target - predicted  # Calculate error

        # Update weights
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += self.learning_rate * error  # Update w0 (bias)
            else:
                self.weights[i] += self.learning_rate * error * inputs[i-1]  # Update other weights

    def calculate_cost(self, inputs, targets):
        # Function to calculate cost (mean squared error)
        predictions = [self.predict(inputs[i]) for i in range(len(inputs))]
        errors = [(predictions[i] - targets[i]) ** 2 for i in range(len(targets))]
        return np.mean(errors)

    def train(self, training_inputs, targets, epochs):
        # Function to train the model for a specific number of epochs
        for epoch in range(epochs):
            # Iterate over each training data and target pair
            for inputs, target in zip(training_inputs, targets):
                # Update weights
                self.update_weights(inputs, target)
            # Calculate and store cost after each epoch
            cost = self.calculate_cost(training_inputs, targets)
            self.costs.append(cost)

    def test(self, test_inputs, targets):
        # Function to test the model with test data and calculate accuracy
        correct = 0  # Initialize correct predictions counter
        total = len(test_inputs)  # Calculate total number of test samples

        print("Testing:")
        for inputs, target in zip(test_inputs, targets):
            # Make a prediction with the model
            prediction = self.predict(inputs)
            # Convert output to binary (0 or 1)
            if prediction >= 0.5:
                output = 1
            else:
                output = 0
            print(f"Input: {inputs}, Target: {target}, Predicted: {output}")

            if output == target:
                # If prediction is correct, increment counter
                correct += 1

        # Calculate error and accuracy on a scale of 0 to 1
        error = self.calculate_cost(test_inputs, targets) / len(test_inputs)  # Mean error
        accuracy = correct / total  # Accuracy

        # Print normalized error and accuracy
        print(f"Mean Error: {error:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

    def plot_costs(self):
        # Function to plot costs during training
        plt.plot(range(1, len(self.costs) + 1), self.costs, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Training Cost vs. Epochs')
        plt.grid(True)
        plt.show()

    def plot_predictions(self, test_inputs, targets):
        # Function to plot model predictions along with expected values
        predictions = [self.predict(inputs) for inputs in test_inputs]
        plt.scatter(range(len(predictions)), predictions, label='Predicted', marker='o')
        plt.scatter(range(len(targets)), targets, label='Actual', marker='x')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Predicted vs. Actual')
        plt.legend()
        plt.grid(True)
        plt.show()


def case_one():

    adaline_model = Adaline(input_size=4, learning_rate=0.05)
    
    inputs = [ [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1],[1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1] ]

    desired_outputs = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]

    adaline_model.train(inputs, desired_outputs, epochs=20)
    adaline_model.test(inputs, desired_outputs)
    adaline_model.plot_costs()


def case_two():
    
    constant = 4
    training_data = []
    test_data = []
    
    training_patterns = []
    training_outputs = []
    
    test_patterns = []
    test_outputs = []
    
    for i in range(1120): 
        
        xt = random.uniform(0, 1)
        eq = constant * xt * (1 - xt)
        
        if i < 19:
            pass
        
        if i >= 19 and i < 1019:
            training_data.append(eq)

        if i >= 1020:
            test_data.append(eq)
            
            
    
    print(len(training_data))
    print(len(test_data))
    
    print("Training Patterns")
    
    for i in range(len(training_data)):
        
        x1 = training_data[i]
        x2 = training_data[i+1]
        x3 = training_data[i+2]
        result = training_data[i+3]
        
        training_patterns.append([x1, x2, x3])
        training_outputs.append(result)
    
    print("Training Patterns")
    print(training_patterns)
    print(training_outputs)


os.system('cls')
case_one()
