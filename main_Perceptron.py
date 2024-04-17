#Importamos librerias
import numpy as np

#Generamos la calse de Perceptron
class Perceptron(object):
    #Inicializamos la matriz de pesos y definimos nuestra constante de aprendizaje
    def __init__(self, N, alpha = 0.2):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    #Función de pasos
    def step(self, x):
        return 1 if x > 0 else 0
    
    #Aqui insertamos una columna de 1's dentro de nuestra matriz de datos de aprendizaje
    # y también definimos la cantidad de epochs por la cual nuestro modelo entrenará
    def fit(self, X, y, epochs = 10):
        X = np.c_[X, np.ones((X.shape[0]))]
        #Ciclo for sobre la cantidad total de epochs
        for epoch in np.arange(0, epochs):
            #Ciclo for por cada punto de datos individual
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                #Aqui en caso de que no se prediga el resultado que se espera, sacamos la diferencia y lo sumamoso al error
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x
    
    #En este método recibimos un set de datos
    def predict(self, X, addBias = True):
        #Revisamos que hayamos recibido una matriz
        X = np.atleast_2d(X)

        #Agregamos una columna de 1's en caso de ser necesario en la matriz
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        
        #Regresamos el producto punto entre la entrada y la matriz de pesos
        return self.step(np.dot(X, self.W))

#Matriz de aprendizaje   
X = np.array([[0,0,0,0],
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
              [1,1,1,1]])

#Salidas esperadas
y = np.array([[0],
              [0],
              [1],
              [0],
              [0],
              [0],
              [1],
              [0],
              [0],
              [0],
              [1],
              [0],
              [1],
              [1],
              [1],
              [1]])

#Llamamos nuestra clase y pasamos los parametros necesarios
p = Perceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs = 20)

#Revisamos las salidas
for (x, target) in zip(X,y):
    pred = p.predict(x)
    print("Data={}, ground-truth={}, pred={}".format(x, target[0], pred))