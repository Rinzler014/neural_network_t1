import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, eta = 0.5, n_inter = 50, random_state = 1):
        self.eta = eta #tasa de aprendizaje
        self.n_inter = n_inter #numero de veces que va a pasar el conjunto de datos completo
        self.random_state = random_state #Semilla del generador de numero aleatorios

    def fit(self, X, y):
        #X Vector de entrenamiento
        #_features = numero de caracteristicas
        #y = vector de etiquetas de respuesta
        rgen = np.random.RandomState(self.random_state) #Generamos numeros aleatorios
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1]) #numero aleatorios con dev 0.01
        #self.w_ = [0,0,0]
        self.errores_ = [] #Lista para errores
        
        print('pesos iniciales', self.w_)

        for _ in range(self.n_inter): #ciclo que se repite segun el numero de iteraciones
            errores = 0
            for xi, etiqueta in zip(X,y):
                actualizacion  = self.eta * (etiqueta - self.predice(xi))
                self.w_[1:] += actualizacion * xi
                self.w_[0] += actualizacion
                errores += int(actualizacion != 0)
            self.errores_.append(errores)
            print('Pesos en epoch', _ , ':', self.w_)
        return self

    def entrada_neta(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predice(self, X):
        return np.where(self.entrada_neta(X) >= 0.0, 1, -1)
        
datos = [[1,1], [1,-1], [-1,1], [-1,-1]]
X = np.array(datos)
y = [1, 1, 1, -1]

print(X)
print(y)

plt.scatter(X[0:3, 0], X[0:3, 1], color = 'red', marker = 'o', label = 'Positivo')
plt.scatter(X[3, 0], X[3, 1], color = 'blue', marker = 'x', label = 'Negativo')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper center')

plt.show()

ppn = Perceptron(eta = 0.5, n_inter = 10)

ppn.fit(X,y)

plt.plot(range(1, len(ppn.errores_) + 1), ppn.errores_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Numero de actualizaciones')
plt.show()