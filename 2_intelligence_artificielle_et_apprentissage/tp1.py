
import numpy as np
import matplotlib.pyplot as plt


K = 5 #max 10 (for the color)
ALLOW_PRINT_DEBUG = True
WITH_GRAPH = False


#describe a point with any number of coordinates
class Point:
    def __init__(self, coords: list[float]):
        self.coords = coords
    
    @staticmethod
    def average(nb_coords, points: list):
        nb_points = len(points)
        if nb_points == 0:
            return points

        result = [0 for i in range(nb_coords)]
        
        for point in points:
            if len(point.coords) != nb_coords:
                print("error.\nIn Point.average().\nThe point has a different number of coordinate then expected.")
                return None #throw error
            for i in range(nb_coords):
                result[i] += point.coords[i]
        
        res = [(val / nb_points) for val in result]
        return res

    @staticmethod
    # return 2 coords set where one has all values as the minimum existing, in points, and the other with the maximum.
    def min_max(nb_coords, points: list):
        min = points[0].coords.copy()
        max = points[0].coords.copy()
        
        for point in points:
            if len(point.coords) != nb_coords:
                #raise error
                print("error.\nIn Point.min_max().\nSome points have different coordinates number.")

            for i, val in enumerate(point.coords):
                if val < min[i]:
                    min[i] = val
                if val > max[i]:
                    max[i] = val

        return min, max

    
    def dist(self, other_point):
        nb_coords = len(self.coords)
        other_coords = other_point.coords
        if len(other_coords) != nb_coords:
            print("error, in dist() in Point.\n Points has different coordinates number.")
        
        squared = []
        for i in range(nb_coords):
            x = (self.coords[i] - other_coords[i]) **2
            squared.append(x)
        
        total = 0
        for nb in squared:
            total += nb

        dist = np.sqrt(total)
        return dist
    
        

#Coordinates set associate with a ummutable group, and a variable cluster
class Flower(Point):
    def __init__(self, group: int, coords: list[int]):
        super().__init__(coords)
        self.group = group
        self.cluster = 0


    def __str__(self):
        return f"group:<{self.group}>, cluster:<{self.cluster}>, <{self.coords}>"
    
    
    #return True if self.cluster has changed
    def update_cluster(self, new_cluster: int) -> bool:
        if self.cluster != new_cluster:
            self.cluster = new_cluster
            return True
        return False


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




















def extract_datas() -> dict:
    datas: list[Point] = []
    
    datas_label = []
    with open("iris_label.csv", "r") as file:
        for i, line in enumerate(file):
            label = int(line.split(";")[0])
            datas_label.append(label)

    with open("iris_data.csv", "r") as file:
        for i, line in enumerate(file):
            temp = line.split(";")[0:-1]
            data = Flower(datas_label[i], [float(val) for val in temp])
            datas.append(data)

    return datas
        


def create_centers(nb, datas: list[Point]) -> list[Cluster]:
    centers: list[Point] = []
    min, max = Point.min_max(len(datas[0].coords), datas)

    for i in range(nb):
        randoms_coords = []
        for j in range(len(datas[0].coords)):
            #... WIP. r must be a float value
            r = np.random.default_rng().integers(low=min[j], high=max[j], size=1)
            randoms_coords.append(r)
        center = Cluster(i+1, randoms_coords)
        centers.append(center)
    return centers



def assign_datas_to_clusters(datas, clusters) -> bool:
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




# return the proportion of the most prensent group if the cluster
def cluster_purity(cluster_id: int, cluster_size: int, datas: list[Flower]) -> float:
    group_concentration = {} #key=Flower.group, value=occurence in datas

    for point in datas:
        if point.cluster != cluster_id:
            continue

        group = point.group
        if group not in group_concentration.keys():
            group_concentration[group] = 1
        else:
            group_concentration[group] += 1

    max = 0
    for key, value in group_concentration.items():
        if value > max:
            max = group_concentration[key]
    
    if max == 0:
        return 0
    return (max / cluster_size) *100





def global_purity(clusters: list[Cluster], datas: list[Flower]) -> float:
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
    purities = {}
    clusters = []

    for i in range(k_max):
        if i < 1:
            continue

        average_purity = 0
        for j in range(100):
            clusters.clear()
            clusters = create_centers(i, datas)
            algo(clusters, datas)
            purity = global_purity(clusters, datas)
            average_purity += purity
            print(f"test {j}. nb clusters = {i}, purity = {purity}")

        average_purity /= 100
        purities[i] = average_purity
        print(f"nb clusters = {i}, average purity = {average_purity}")

    best = 0
    greater = 0
    for key, value in purities.items():
        if ALLOW_PRINT_DEBUG:
            print(f"k={key}, purity={value}.")
        if value > greater:
            greater = value
            best = key

    return best




def main():
    datas:list[Flower] = extract_datas()

    if not WITH_GRAPH:
        better_k = found_better_k(datas, K)
        print(f"Better nb of clusters (greater purity): {better_k}")
        return


    #WITH GRAPH (algo is running only one time)
    centers:list[Cluster] = create_centers(K, datas)

    # clusters' center at the start
    plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="black", s=100)
    plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="yellow", s=50)

    algo(centers, datas)

    purity = global_purity(centers, datas)
    if ALLOW_PRINT_DEBUG:
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


    print("\nExternal color: cluster\nInternal color: flower group\n!! Dot black+yellow = cluster's center")
           

    colors = ["red", "green", "blue", "yellow", "pink", "orange", "purple", "beige", "brown", "gray"]
    i = 0
    
    #flower
    for key, points in datas_sorted_by_cluster.items():
        plt.scatter([point.coords[0] for point in points], [point.coords[1] for point in points], color=colors[i])
        i += 1


    #flower's group
    colors_flowers = {0: "red", 1: "green", 2: "blue"}
    for point in datas:
        plt.scatter([point.coords[0]], [point.coords[1]], color=colors_flowers[point.group], s=5)
    #plt.scatter([point.coords[0] for point in datas], [point.coords[1] for point in datas], color=colors[i], s=5)
    

    #clusters' center
    plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="black")
    plt.scatter([point.coords[0] for point in centers], [point.coords[1] for point in centers], color="yellow", s=10)

    plt.show()







if __name__ == "__main__":
    main()

