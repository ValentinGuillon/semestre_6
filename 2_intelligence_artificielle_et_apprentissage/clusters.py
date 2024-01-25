import numpy as np
import matplotlib.pyplot as plt

# On va ici utiliser des fonctions de la librairie NumPy
# (www.numpy.org) pour générer aléatoirement des données, autour d'un
# centre. Il s'agit de données générées à partir d'une distribution
# normale, soit d'une gaussienne. Les paramètres fournis à la fonction
# sont le centre, c'est à dire le point autour duquel les autres
# seront répartis, soit la moyenne de la gaussienne ; la covariance :
# la matrice diagonale d'étalement des éléments ; ainsi que le nombre
# de points à générer. Si on veut enlever des clusters, on peut
# commenter, si on veut en ajouter, on peut copier-coller (en faisant
# attention aux centres):
# C1 ---------->
centre1=np.array([3,3])            # centre du premier cluster
sigma1=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du premier cluster
taille1=250                        # nombre de points du premier cluster
cluster1=np.random.multivariate_normal(centre1,sigma1,taille1)
# C1 <----------
# C2 ---------->
centre2=np.array([-3,-3])          # centre du second cluster
sigma2=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du second cluster
taille2=200                        # nombre de points du second cluster
cluster2=np.random.multivariate_normal(centre2,sigma2,taille2)
# C2 <----------
# C3 ---------->
centre3=np.array([-3,3])          # centre du troisieme cluster
sigma3=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du second cluster
taille3=300                        # nombre de points du second cluster
cluster3=np.random.multivariate_normal(centre3,sigma3,taille3)
# C3 <----------
# C4 ---------->
centre4=np.array([3,-3])          # centre du quatrieme cluster
sigma4=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du second cluster
taille4=150                        # nombre de points du second cluster
cluster4=np.random.multivariate_normal(centre4,sigma4,taille4)
# C4 <----------


clusters=np.concatenate((cluster1, cluster2, cluster3, cluster4))

# On utilise ici les fonctions de MatPlotLib (matplotlib.org) pour
# afficher les points.  On commence par générer les listes d'abcisses,
# puis d'ordonnées pour chaque cluster.  Ensuite, on ajouter les
# points aux clusters (NB : si on voulait distinguer les clusters dès
# le départ, on pourrait spécifier les couleurs) :
plt.scatter([point[0] for point in clusters], [point[1] for point in clusters], color="black")
plt.scatter([point[0] for point in cluster1], [point[1] for point in cluster1], color="pink")
plt.scatter([point[0] for point in cluster2], [point[1] for point in cluster2], color="blue")
plt.scatter([point[0] for point in cluster3], [point[1] for point in cluster3], color="green")
plt.scatter([point[0] for point in cluster4], [point[1] for point in cluster4], color="yellow")

# Si on veut afficher les vrais centres :
plt.scatter(centre1[0], centre1[1], color="red") # coloration en rouge du centre du cluster1
plt.scatter(centre2[0], centre2[1], color="red") # coloration en rouge du centre du cluster2
plt.scatter(centre3[0], centre3[1], color="red") # coloration en rouge du centre du cluster2
plt.scatter(centre4[0], centre4[1], color="red") # coloration en rouge du centre du cluster2
plt.show()

