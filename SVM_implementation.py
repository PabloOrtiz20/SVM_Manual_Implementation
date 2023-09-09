import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# Iniciamos seteando los valores del learning_rate, C y epochs
# En este caso seleccionamos un learning_rate muy bajo ya que el algoritmo converge rápido
learning_rate = 0.000001 
C = 0.01
epochs = 1000

# Utilizamos la función de make_blobs de sklearn para generar un dataset con las clases
X,y = datasets.make_blobs(
    n_samples=1000, n_features=2, centers = 2, cluster_std=3)

# Separamos la data en set de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Usando el data set de entrenamiento, sacamos el número de parametros e instancias
n_samples, n_features = X_train.shape

# Transformamos las variables para que todas las etiquetas tengan -1 o 1 en vez de 0 o 1
y_transformed = np.where(y_train <= 0,-1,1)

# Idealmente iniciaríamos con "weights" y "bias" randomizados

w = np.random.randn(n_features)
b = np.random.randn()

# Creamos una lista vacía donde guardaremos los valores de accuracy después de cada época
accuracies = []

# Con el valor establecido al inicio del código, iteramos sobre el número de épocas
for x in range(epochs):

  # Usamos la función "enumerate" para obtener el índice de cada instancia 
  # e iteramos sobre cada una
  for idx, x_i in enumerate(X_train):
      
    # Para cada iteración creamos una variable "condition" que determinará 
    # si la clase objetivo de dicha instancia, al ser multiplicada por los pesos 
    # y agregada al bias, es mayor o igual a 1 (es decir que pertenece a la clase 1)
    condition = y_transformed[idx] * (np.dot(x_i,w)-b) >= 1
    
    # Si la instancia pertenece a dicha clase se ajusta el peso 
    if condition: 
      w -= learning_rate * (2 * C * w)
      
    # En caso de que no pertenezca se ajusta el peso y el bias
    else: 
      w -= learning_rate * (2 * C * w - np.dot(x_i, y_transformed[idx]))
      b -= learning_rate * y_transformed[idx]
     
  # Se calculan las clases con el nuevo peso y el nuevo bias 
  # y se calcula el accuracy para esa época
  approx_in = np.dot(X_train, w) - b
  y_pred_in = np.sign(approx_in)
  accuracy_in = np.sum(y_transformed == y_pred_in) / len(y_transformed)
  accuracies.append(accuracy_in)
  print("Epoch {} | Accuracy {}".format(x+1,accuracy_in))

# Al terminar las épocas, se calculan las nuevas aproximaciones usando el test set
approx = np.dot(X_test, w) - b
y_pred = np.sign(approx)
y_test_transformed = np.where(y_test <= 0,-1,1)

# Calculamos el accuracy final
accuracy = np.sum(y_test_transformed == y_pred) / len(y_test_transformed)
print("SVM classification accuracy", accuracy)


def hiperplano_grafica(x, w, b, offset):
    # Creamos una función llamada "hiperplano_grafica" para hacer el plot de las 
    # líneas en el gráfico de los vectores
    return (-w[0] * x + b + offset) / w[1]

# Creamos la figura y gráficamos los puntos con los colores de las clases
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y,cmap="cividis")

# Se calculan los valores x0_1 y x0_2 que se utilizan para determinar el rango 
# del eje x al trazar el límite de decisión del clasificador
x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

# Se calculan los valores x1_1 y x1_2 para graficar la línea punteada del clasificador
x1_1 = hiperplano_grafica(x0_1, w, b, 0)
x1_2 = hiperplano_grafica(x0_2, w, b, 0)

# Se calculan x1_1_m y x1_2_m para la línea inferior
x1_1_m = hiperplano_grafica(x0_1, w, b, -1)
x1_2_m = hiperplano_grafica(x0_2, w, b, -1)

# Se calculan x1_1_p y x1_2_p para la línea superior
x1_1_p = hiperplano_grafica(x0_1, w, b, 1)
x1_2_p = hiperplano_grafica(x0_2, w, b, 1)

# Graficamos las líneas
ax.plot([x0_1, x0_2], [x1_1, x1_2], "k--")
ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

# Calculamos los límites de las gráficas usando el mínimo y el máximo del dataset
x1_min = np.amin(X[:, 1])
x1_max = np.amax(X[:, 1])
ax.set_ylim([x1_min - 3, x1_max + 3])

# Agregamos el título y mostramos la gráfica
plt.title("SVM classification accuracy {}".format(accuracy))
plt.show()

# Gráfica de la precisión a lo largo de las épocas
plt.figure()
plt.plot(range(1, epochs + 1), accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt

# Calculamos la matriz de confusión
confusion = confusion_matrix(y_test_transformed, y_pred)

# Creamos la gráfica de la matriz
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[-1, 1])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Calculamos el reporte de clasificación completo con sklearn
class_report = classification_report(y_test_transformed, y_pred)

# Imprimimos el reporte de clasificación completo en la consola
print("Classification Report:\n", class_report)

# Extraemos las métricas del reporte de clasificación
class_names = ['Class -1', 'Class 1']
precision = [float(class_report.split()[i]) for i in range(5, 11, 5)]
recall = [float(class_report.split()[i]) for i in range(6, 12, 5)]
f1_score = [float(class_report.split()[i]) for i in range(7, 13, 5)]

# Creamos la figura
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.2
index = np.arange(len(class_names))

# Creamos cada una de las barras
bar1 = plt.bar(index, precision, bar_width, label='Precision', alpha=0.8)
bar2 = plt.bar(index + bar_width, recall, bar_width, label='Recall', alpha=0.8)
bar3 = plt.bar(index + 2 * bar_width, f1_score, bar_width, label='F1-Score', alpha=0.8)

# Creamos las etiquetas y la leyenda
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Classification Metrics by Class')
plt.xticks(index + bar_width, class_names)
plt.legend()

plt.tight_layout()
plt.show()