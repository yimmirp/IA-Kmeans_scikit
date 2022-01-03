import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X=np.array([ [8,2],[9,7],[2,12],[9,1],[10,7],[3,14],[8,1],[1,13]])

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.cluster_centers_)
plt.title('Kmeans con SCIKIT')
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow', label = 'Centroids')
plt.xlabel('Efectividad')
plt.ylabel('PH')
print("Clases")
print(kmeans.predict(X))
cadena=""
for i , j in enumerate(kmeans.labels_):
    print("Punto({0}) Clase: {1}".format(X[i],j))
    cadena+="Punto({0}) Clase: {1}".format(X[i],j)+"   "
    
plt.xlabel(cadena)
plt.show()