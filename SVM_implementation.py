import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("darkgrid")

#Seleccionamos un learning rate bajo ya que el algoritmo converge muy rápido 
#y queremos mostrar la mejora a través del tiempo.

learning_rate = 0.000001 
C = 0.01
epochs = 1000

#Utilizamos la función de make_blobs de sklearn para generar un dataset para separar.
#Como queremos replicar los resultados en las corridas del código, usaremos random_state.

X,y = datasets.make_blobs(
    n_samples=1000, n_features=2, centers = 2, cluster_std=5, random_state=40)

#Separamos la data en set de entrenamiento y prueba y de nuevo usaremos un random_state 
#para replicar resultados. 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=210)

n_samples, n_features = X_train.shape
y_transformed = np.where(y_train <= 0,-1,1)

#Idealmente iniciaríamos con "weights" y "bias" randomizados, pero en este caso usaremos 
#valores fijos para replicar resultados. 

#w = np.random.randn(n_features)
#b = np.random.randn()

w = np.zeros(n_features)
b = 0.35

accuracies = []
for x in range(epochs):
    
  for idx, x_i in enumerate(X_train):
    condition = y_transformed[idx] * (np.dot(x_i,w)-b)>= 1
    
    if condition: 
      w -= learning_rate * (2 * C * w)
    else: 
      w -= learning_rate * (2 * C * w - np.dot(x_i, y_transformed[idx]))
      b -= learning_rate * y_transformed[idx]
     
  approx_in = np.dot(X_train, w) - b
  y_pred_in = np.sign(approx_in)
  accuracy_in = np.sum(y_transformed == y_pred_in) / len(y_transformed)
  accuracies.append(accuracy_in)
  print("Epoch {} | Accuracy {}".format(x+1,accuracy_in))
      
approx = np.dot(X_test, w) - b

y_pred = np.sign(approx)

y_test_transformed = np.where(y_test <= 0,-1,1)

accuracy = np.sum(y_test_transformed == y_pred) / len(y_test_transformed)
print("SVM classification accuracy", accuracy)

def hiperplano_grafica(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y,cmap="cividis")

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

x1_1 = hiperplano_grafica(x0_1, w, b, 0)
x1_2 = hiperplano_grafica(x0_2, w, b, 0)

x1_1_m = hiperplano_grafica(x0_1, w, b, -1)
x1_2_m = hiperplano_grafica(x0_2, w, b, -1)

x1_1_p = hiperplano_grafica(x0_1, w, b, 1)
x1_2_p = hiperplano_grafica(x0_2, w, b, 1)

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k--")
ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

x1_min = np.amin(X[:, 1])
x1_max = np.amax(X[:, 1])
ax.set_ylim([x1_min - 3, x1_max + 3])

plt.title("SVM classification accuracy {}".format(accuracy))
plt.show()