
import numpy as np
import matplotlib.pyplot as plt


K = 4 #max 10 (for the color)
DATAS_SRC = "iris" #"iris" or "random"
WITH_TESTS = True #with True, test x times the algo for 1 to K clusters.
WITH_GRAPH = True #with True, one time alog then graph.
DIST_METHOD = "euclidian" #"manhattan" or "euclidian" or "minkowski"
MINKOWSKI_VALUE = 3 #1=> manhattan, 2=> euclidian

ALLOW_PRINT_DEBUG = False
SHOW_CENTER_INIT_POS = False



#describe a point with any number of coordinates, associate to a cluster, and an optional group (to calculate cluster's purity)
class Point:
    pass #just to use its name inside itself
class Point:
    def __init__(self, coords: list[float], group: any = 0):
        self.coords = coords
        self.cluster = -1 #is's cluster this point is attached

        self.group = group


    def __str__(self):
        return f"group:<{self.group}>, cluster:<{self.cluster}>, <{self.coords}>"


    @staticmethod
    def average(nb_coords, points: list[Point]) -> list[float]:
        nb_points = len(points)
        if nb_points == 0:
            return points

        results = [0 for _ in range(nb_coords)]
        
        for point in points:
            if len(point.coords) != nb_coords:
                print("error.\nIn Point.average().\nThe point has a different number of coordinate then expected.")
                return None #throw error
            for i in range(nb_coords):
                results[i] += point.coords[i]
        
        final_res = [(val / nb_points) for val in results]
        return final_res


    #return True if self.cluster has changed
    def update_cluster(self, new_cluster: int) -> bool:
        if self.cluster != new_cluster:
            self.cluster = new_cluster
            return True
        return False


    
    def dist_euclidian(self, other_point):
        return self.dist_minkowski(other_point, 2)


    def dist_manhattan(self, other_point):
        return self.dist_minkowski(other_point, 1)
    
    
    #dist minkowski
    def dist(self, other_point, p = 0):
        #p=1 => Manhattan, p=2 => euclidian
        if p <= 0:
            if DIST_METHOD == "manhattan":
                p = 1
            elif DIST_METHOD == "euclidian":
                p = 2
            elif DIST_METHOD == "minkowski":
                p = MINKOWSKI_VALUE
    
        nb_coords = len(self.coords)
        other_coords = other_point.coords
        if len(other_coords) != nb_coords:
            print("error, in dist() in Point.\n Points has different coordinates number.")
        
        squared = []
        for i in range(nb_coords):
            x = np.absolute(self.coords[i] - other_coords[i]) ** p
            squared.append(x)
        
        total_sum = 0
        for nb in squared:
            total_sum += nb

        dist = total_sum ** (1/p)
        return dist
        

# #Coordinates set associate with a ummutable group, and a variable cluster
# class Flower(Point):
#     def __init__(self, group: int, coords: list[int]):
#         super().__init__(coords)
#         self.group = group
#         self.cluster = 0


#     def __str__(self):
#         return f"group:<{self.group}>, cluster:<{self.cluster}>, <{self.coords}>"
    
    
#     #return True if self.cluster has changed
#     def update_cluster(self, new_cluster: int) -> bool:
#         if self.cluster != new_cluster:
#             self.cluster = new_cluster
#             return True
#         return False


#Coordinates set associate with a immutable id, and a variable nb of point set to this cluster
class Cluster(Point):
    def __init__(self, id: int, coords: list[int]):
        super().__init__(coords)
        self.id = id
        self.nb_points = 0
        
    def __str__(self):
        res = f"id:<{self.id}>, points:<{self.nb_points}>, <["
        for i in range(len(self.coords)-1):
            res += f"{self.coords[i]:.2f}, "
        res += f"{self.coords[-1]:.2f}]>"
            
        return res
    
    # return the average of the distance between all possible data, set to this cluster
    def similarity_intra(self, datas: list[Point]) -> float:
        cluster_points = [point for point in datas if point.cluster == self.id]

        total = 0

        for i, p1 in enumerate(cluster_points[0: -1]):
            for p2 in cluster_points[i+1: -1]:
                total += p1.dist(p2)
        
        return total







def SSE(clusters: list[Cluster], datas: list[Point]) -> float:
    res = 0.0
    for cluster in clusters:
        ssec = cluster.similarity_intra(datas)
        res += ssec
    
    return res














def extract_datas(file_datas: str, file_groups: str = "") -> dict:
    datas: list[Point] = []
    # with_group = True
    # if not file_groups:
    #     with_group = False
    
    datas_label = []
    with open(file_groups, "r") as file:
        for i, line in enumerate(file):
            label = int(line.split(";")[0])
            datas_label.append(label)

    with open(file_datas, "r") as file:
        for i, line in enumerate(file):
            temp = line.split(";")[0:-1]
            data = Point([float(val) for val in temp], datas_label[i])
            datas.append(data)

    return datas
        

def generate_datas() -> dict:
    #coded by ARA (except: to_ret =...)
    # C1 ---------->
    centre1=np.array([3,3])            # centre du premier cluster
    sigma1=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du premier cluster
    taille1=25                        # nombre de points du premier cluster
    cluster1=np.random.multivariate_normal(centre1,sigma1,taille1)
    # C1 <----------
    # C2 ---------->
    centre2=np.array([-3,-3])          # centre du second cluster
    sigma2=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du second cluster
    taille2=25                        # nombre de points du second cluster
    cluster2=np.random.multivariate_normal(centre2,sigma2,taille2)
    # C2 <----------
    # C3 ---------->
    centre3=np.array([-3,3])          # centre du troisieme cluster
    sigma3=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du second cluster
    taille3=25                        # nombre de points du second cluster
    cluster3=np.random.multivariate_normal(centre3,sigma3,taille3)
    # C3 <----------
    # C4 ---------->
    centre4=np.array([3,-3])          # centre du quatrieme cluster
    sigma4=np.array([[1.5,0],[0,1.5]]) # matrice de covariance du second cluster
    taille4=25                        # nombre de points du second cluster
    cluster4=np.random.multivariate_normal(centre4,sigma4,taille4)
    # C4 <----------


    clusters=np.concatenate((cluster1, cluster2, cluster3, cluster4))
    to_ret = [Point(coords, 0) for coords in clusters]
    
    return to_ret





def create_centers(nb, datas: list[Point]) -> list[Cluster]:
    centers: list[Point] = []

    # choose randomly an existing point
    for i in range(nb):
        index = np.random.randint(0, len(datas))
        copied_coords = datas[index].coords.copy()
        center = Cluster(i+1, copied_coords)
        centers.append(center)

    return centers




def assign_datas_to_clusters(datas: list[Point], clusters: list[Cluster]) -> bool:
    a_data_changed_of_cluster = False

    #reset nb of points of clusters
    for cluster in clusters:
        cluster.nb_points = 0

    for data in datas:
        min_dist = -1
        nearest_cluster = None
       
        for cluster in clusters:
            dist = data.dist(cluster)
            if min_dist == -1 or min_dist > dist:
                min_dist = dist
                nearest_cluster = cluster
       
        nearest_cluster.nb_points += 1
        temp = data.update_cluster(nearest_cluster.id)
        if temp:
            a_data_changed_of_cluster = temp
    
    return a_data_changed_of_cluster



def update_clusters_coords(datas: list[Point], clusters: list[Cluster]):
    datas_sorted_by_cluster = {}
    
    # sort datas
    for point in datas:
        cluster_id = point.cluster
        if cluster_id not in datas_sorted_by_cluster.keys():
            datas_sorted_by_cluster[cluster_id] = [point]
        else:
            datas_sorted_by_cluster[cluster_id].append(point)

    #update clusters
    for cluster in clusters:
        if cluster.id not in datas_sorted_by_cluster.keys():
            continue
        cluster.coords = Point.average(len(cluster.coords), datas_sorted_by_cluster[cluster.id])




# return the proportion of the most present group in the cluster
def cluster_purity(cluster_id: int, cluster_size: int, datas: list[Point]) -> float:
    group_concentration = {} #key=Point.group, value=occurence in datas

    #count
    for point in datas:
        if point.cluster != cluster_id:
            continue

        group = point.group
        if group not in group_concentration.keys():
            group_concentration[group] = 1
        else:
            group_concentration[group] += 1

    #search the greater proportion
    max = 0
    for key, value in group_concentration.items():
        if value > max:
            max = group_concentration[key]
    
    # return
    if max == 0:
        return 0
    return (max / cluster_size) *100





def global_purity(clusters: list[Cluster], datas: list[Point]) -> float:
    purity = 0

    for cluster in clusters:
        temp = cluster_purity(cluster.id, cluster.nb_points, datas)
        if ALLOW_PRINT_DEBUG:
            print(f"cluster id: {cluster.id}, purity= {temp}%.")
        purity += temp
    
    return purity / len(clusters)






def algo(clusters: list[Cluster], datas: list[Point]):
    loop = 0
    while True:
        loop += 1

        a_data_changed_of_cluster = assign_datas_to_clusters(datas, clusters)
        update_clusters_coords(datas, clusters)

        if not a_data_changed_of_cluster:
            break

    
def found_better_k(datas: list[Point], k_max: int):
    nb_tests = 10
    purities = {}
    clusters = []
    sses = {}

    for i in range(k_max+1):
        if i < 1:
            continue

        average_purity = 0
        average_sse = 0
        for j in range(nb_tests):
            clusters.clear()
            clusters = create_centers(i, datas)
            algo(clusters, datas)
            purity = global_purity(clusters, datas)
            sse = SSE(clusters, datas)
            average_purity += purity
            average_sse += sse
            if ALLOW_PRINT_DEBUG:
                print(f"test {j}. nb clusters = {i}, purity = {purity}, sse = {sse}")

        average_purity /= nb_tests
        average_sse /= nb_tests
        purities[i] = average_purity
        sses[i] = average_sse
        if ALLOW_PRINT_DEBUG:
            print(f"nb clusters = {i}, average purity = {average_purity}, av sse = {average_sse}")

    best = 0
    greater = 0
    for key, value in purities.items():
        print(f"k={key}, purity={value}, sse = {sses[key]}.")
        if value > greater:
            greater = value
            best = key
    
    plt.clf()
    for key, val in sses.items():
        plt.scatter([key], [val], color="black", s=50)
    
    plt.show()
    plt.clf()

    return best










def main():
    print("The status of the globals are the following:")
    print("K =", K)
    print("DATAS_SRC =", DATAS_SRC)
    print("WITH_GRAPH =", WITH_GRAPH)
    print("DIST_METHOD =", DIST_METHOD)
    print("MINKOWSKI_VALUE =", MINKOWSKI_VALUE)
    print()
    print("ALLOW_PRINT_DEBUG =", ALLOW_PRINT_DEBUG)
    print("SHOW_CENTER_INIT_POS =", SHOW_CENTER_INIT_POS)
    
    input("\nPress Enter to proceed...")
    


    datas:list[Point] = []
    if DATAS_SRC == "iris":
        datas = extract_datas("iris_data.csv", "iris_label.csv")
    elif DATAS_SRC == "random":
        datas = generate_datas()

    if WITH_TESTS:
        better_k = found_better_k(datas, K)
        print(f"Better nb of clusters (greater purity): {better_k}")
    
    if not WITH_GRAPH:
        return


    #WITH GRAPH (algo is running only one time)
    centers:list[Cluster] = create_centers(K, datas)

    # clusters' center at the start
    if SHOW_CENTER_INIT_POS:
        plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="black", s=100, label="Center (init)")
        plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="yellow", s=50)

    algo(centers, datas)

    if ALLOW_PRINT_DEBUG and DATAS_SRC == "iris":
        purity = global_purity(centers, datas)
        print(f"Global purity = {purity}.")



    #AFFICHAGE

    datas_sorted_by_cluster = {}
    # sort datas
    for point in datas:
        cluster_id = point.cluster
        if cluster_id not in datas_sorted_by_cluster.keys():
            datas_sorted_by_cluster[cluster_id] = [point]
        else:
            datas_sorted_by_cluster[cluster_id].append(point)


    print("\nExternal color: cluster\nInternal color: flower group\n!!The graph is only a REPRESENTATION. It may be weird for datas with more than 2 coordinates.")
           

    colors = ["red", "green", "blue", "yellow", "pink", "orange", "purple", "beige", "brown", "gray"]
    i = 0
    
    #datas
    for key, points in datas_sorted_by_cluster.items():
        plt.scatter([point.coords[0] for point in points], [point.coords[1] for point in points], color=colors[i])
        i += 1


    #flower's groups
    if DATAS_SRC == "iris":
        colors_flowers = {0: "red", 1: "green", 2: "blue"}
        for point in datas:
            plt.scatter([point.coords[0]], [point.coords[1]], color=colors_flowers[point.group], s=5)
    
    

    #clusters' center
    plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="black", marker="X", label="Cluster's center (final)")
    plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="yellow", marker="x")

    plt.legend()
    plt.show()

    #plt.clf() #clear the graph
    #plt.pause(0.1)






if __name__ == "__main__":
    main()

