# Robert Vetter
# Lösungenverfahren zfür das TSP

# notwendigen Bibliotheken importieren
from math import sqrt
import random
import networkx as nx
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from itertools import combinations
import time

# Berechnung des Abstands zwischen den beiden Punkten mittels Pythagoras
def distance(point1, point2):
    x1, y1 = point1  # Koordinaten des ersten Punkts
    x2, y2 = point2  # Koordinaten des zweiten Punkts
    return sqrt((x2 - x1)**2 + (y2 - y1)**2) 

# Gesamtlänge Route, Summe der Distanzen zwischen je zwei aufeinanderfolgenden Punkten
def route_length(route):
    return round(sum(distance(route[i], route[i+1]) for i in range(len(route) - 1)), 2) 

# Greedy Algorithmus
def greedy(points):
    unvisited = set(points)  # unbesuchte Punkte
    current_point = random.choice(points)  # aktuellen Punkt zufällig auswählen
    unvisited.remove(current_point)  # aktuellen Punkt aus den unbesuchten Punkten entfernen

    tour = [current_point]  # initiale Tour mit dem aktuellen Punkt

    while unvisited:  # Solange es noch unbesuchte Punkte gibt
        next_point = min(unvisited, key=lambda x: distance(current_point, x))  # nächsten Punkt finden, der am nächsten zum aktuellen Punkt liegt
        tour.append(next_point)  # nächsten Punkt zur Tour hinzufügen
        current_point = next_point  # aktuellen Punkt auf den soeben gefundenen Punkt setzen
        unvisited.remove(current_point)  # aktuellen Punkt aus den unbesuchten Punkten entfernen

    tour.append(tour[0])  # Rückkehr zum Startpunkt

    return tour  # Tour zurückgeben



# Genetischer Algorithmus
 # Fitness der Route, basierend auf der Gesamtstrecke (ähnlich zu Heuristik)
def fitness(route, points):
    return route_length([points[i] for i in route]) 

# Zufällige Route erstellen
def random_route(points):
    return random.sample(range(len(points)), len(points))

# Kreuzen zweier Routen (aus Route 1 wird Routenbruchteil entnommen, wird mit Bruchteil aus anderer Route kombiniert)
def crossover(parent1, parent2):
    slice_point1 = random.randint(0, len(parent1) - 1)  # Zufälliger Index in parent1
    slice_point2 = random.randint(0, len(parent2) - 1)  # Zufälliger Index in parent2
    slice_point1, slice_point2 = min(slice_point1, slice_point2), max(slice_point1, slice_point2)  # Reihenfolge der Indizes korrigieren
    
    child1 = parent1[slice_point1:slice_point2]  # erster Teil des Kindes von parent1
    child2 = []
     # zweiter Teil des Kindes von parent2, der nicht in child1 vorkommt
    for item in parent2:
        if item not in child1:
            child2.append(item)
    child1.extend(child2)  # Kind vollständig zusammensetzen
    
    return child1

# zufälliges Vertauschen zweier Puntke in Route
def mutate(route):
    idx1, idx2 = random.sample(range(len(route)), 2)  # zwei zufällige Indizes auswählen
    route[idx1], route[idx2] = route[idx2], route[idx1]  # Elemente an den ausgewählten Indizes tauschen

def genetic(points, pop_size=150, generations=150, top_percent=10):
    population = [random_route(points) for _ in range(pop_size)]  # Anfangspopulation erstellen
    
    for generation in range(generations):  # Für jede Generation
        population.sort(key=lambda x: fitness(x, points))  # Population nach Fitness sortieren
        new_population = population[:2]  # Zwei besten Routen der neuen Population hinzufügen
        
        while len(new_population) < pop_size:  # Bis die neue Population voll ist
            parent1 = random.choice(population[:top_percent])  # Einen Elternteil aus den top_percent besten Routen auswählen
            parent2 = random.choice(population[:top_percent])  # Einen zweiten Elternteil auswählen
            child = crossover(parent1[:], parent2[:])  # Kind erzeugen und der neuen Population hinzufügen
            new_population.append(child)
        
        for route in new_population[2:]:  # Für jede Route in der neuen Population, außer den zwei besten
            if random.random() < 0.1:  # Mit 10% Wahrscheinlichkeit
                mutate(route)  # Route mutieren
        
        population = new_population  # Neue Population zur aktuellen Population machen
    
    population.sort(key=lambda x: fitness(x, points))  # Letzte Population nach Fitness sortieren
    best_route = population[0]  # Beste Route auswählen
    tour = [points[i] for i in best_route]
    tour.append(tour[0]) # Anfangsknoten hinzufügen
    return tour



def mst_dfs(points):
    # leeren Graphen erstellen
    G = nx.Graph()

    # Füge alle Punkte als Knoten hinzu
    for point in points:
        G.add_node(point)

    # Füge die Kanten mit ihren Entfernungen hinzu
    for p1, p2 in combinations(points, 2):
        G.add_edge(p1, p2, weight=distance(p1, p2))

    # Erstelle den MST
    mst = nx.minimum_spanning_tree(G)

    # DFS um den Pfad zu erstellen
    tour = list(nx.dfs_preorder_nodes(mst, source=points[0]))

    # Startpunkt zur Route hinzufügen
    tour.append(tour[0])

    return tour

# vertauscht zwei Kanten, um Routenlänge zu minimieren
def two_opt(route):
    best = route  # initiale Route
    best_length = route_length(best)  # Länge der initialen Route
    improved = True  # Variable für Verbesserungen
    
    while improved: # solange noch kürzere Routen vorhanden
        improved = False  # zurücksetzen von improved
        for i in range(1, len(route) - 2):  # gehe durch alle Knoten
            for j in range(i + 1, len(route)):  
                if j - i == 1: continue  # überspringe benachbarte Knoten
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]  # tausche Knoten
                
                new_length = route_length(new_route)  # berechne neue Länge
                
                if new_length < best_length:  # wenn kürzer, update best
                    best = new_route
                    best_length = new_length
                    improved = True
        route = best  # setze Route auf best
    return best  # gebe beste Route zurück


def plot_routes(coordinates, routes, names, times):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Flattening, um leicht darüber zu iterieren
    axs = axs.flatten()

    # Plot der Punkte für jeden Subplot
    for ax in axs:
        x_coo, y_coo = zip(*coordinates)
        ax.scatter(x_coo, y_coo, c='black')

    # Zeichnet jede Route in ihrem eigenen Subplot
    for i, (route, name, ax, exec_time) in enumerate(zip(routes, names, axs, times)):
        x = [p[0] for p in route]
        y = [p[1] for p in route]
        ax.plot(x, y, '-')
        ax.set_title(f"{name}: {route_length(route)} (Time: {exec_time:.4f}s)")

    plt.show()


if __name__ == "__main__":
    # Parameter für Genetic
    pop_size = 2000;
    num_generations = 200;
    mutation_rate = 10;


    num_points = int(input("Bitte geben Sie die Anzahl der Punkte ein: "))
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]
    pointsnp = np.array(points);

    # Greedy-Algorithmus
    start_time = time.time()
    greedy_route = greedy(points)
    greedy_time = time.time() - start_time
    print(f"Greedy-Länge: {route_length(greedy_route)}, Zeit: {greedy_time:.4f}s")

    # Genetic
    start_time = time.time()
    genetic_route = genetic(points, pop_size, num_generations, mutation_rate)
    genetic_time = time.time() - start_time
    print(f"Genetic-Länge: {route_length(genetic_route)}, Zeit: {genetic_time:.4f}s")

    # MST-DFS
    start_time = time.time()
    mst_dfs_route = mst_dfs(points)
    mst_dfs_time = time.time() - start_time
    print(f"MST-Länge: {route_length(mst_dfs_route)}, Zeit: {mst_dfs_time:.4f}s")

    # Zeitmessungen
    start_time = time.time()
    greedy_opt_route = two_opt(greedy_route)
    greedy_opt_time = time.time() - start_time

    start_time = time.time()
    genetic_opt_route = two_opt(genetic_route)
    genetic_opt_time = time.time() - start_time

    start_time = time.time()
    mst_dfs_opt_route = two_opt(mst_dfs_route)
    mst_dfs_opt_time = time.time() - start_time

    all_routes = [greedy_route, genetic_route, mst_dfs_route, 
              greedy_opt_route, genetic_opt_route, mst_dfs_opt_route]

    all_names = ['Greedy', 'Genetic', 'MST-DFS', 
             'Greedy Optimized', 'Genetic Optimized', 'MST-DFS Optimized']

    all_times = [greedy_time, genetic_time, mst_dfs_time, 
             greedy_opt_time, genetic_opt_time, mst_dfs_opt_time]

    plot_routes(pointsnp, all_routes, all_names, all_times)
