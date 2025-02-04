import networkx as nx
import matplotlib.pyplot as plt

# Function to generate the Collatz graph for a given number
def generate_collatz_graph(n):
    G = nx.DiGraph()
    while n != 1:
        if n % 2 == 0:
            next_n = n // 2
        else:
            next_n = 3 * n + 1
        G.add_edge(n, next_n)
        n = next_n
    return G

# Function to visualize the graph
def visualize_collatz_path(n):
    G = generate_collatz_graph(n)
    
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, edge_color="gray", node_color="lightblue")
    plt.title(f"Collatz Graph for {n}")
    plt.show()

# Example: Visualize the Collatz path for 987654
visualize_collatz_path(987654)
