import heapq

# Function to generate the next possible states in the Collatz sequence
def collatz_neighbors(n):
    neighbors = []
    if n % 2 == 0:
        neighbors.append(n // 2)  # Even step
    else:
        neighbors.append(3 * n + 1)  # Odd step
    return neighbors

# Dijkstra's Algorithm for shortest path to 1
def collatz_pathfinding(start):
    priority_queue = [(0, start)]  # (steps, number)
    visited = {}
    
    while priority_queue:
        steps, num = heapq.heappop(priority_queue)
        
        if num == 1:
            return steps  # Found the shortest path to 1
        
        if num in visited:
            continue
        visited[num] = steps
        
        for neighbor in collatz_neighbors(num):
            heapq.heappush(priority_queue, (steps + 1, neighbor))

# Example: Find the shortest Collatz path for 987654
test_number = 987654
shortest_path_steps = collatz_pathfinding(test_number)
print(f"Shortest path to 1 for {test_number}: {shortest_path_steps} steps")
