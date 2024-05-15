# PacmanAI- Search Algorithms Implementation

Project Link :
http://ai.berkeley.edu/project_overview.html

## Introduction
This project aims to implement search algorithms to assist Pac-Man in navigating maze-like environments efficiently. The objective is to develop general search algorithms and apply them to various Pac-Man scenarios, enabling Pac-Man to find paths to specific locations and collect food optimally.

## How to Use
1. **Cloning the Repository:**
    - Clone the assignment repository using the provided GitHub classroom link.
    - Open a terminal and run the following command:
      ```
      git clone https://github.com/shakir-flash/Pacman-AI-Search-Algorithms.git
      ```

2. **Running Pac-Man:**
    - Navigate to the cloned directory.
    - Change directory to the `search` folder.
    - Run Pac-Man using the following command:
      ```python
      python pacman.py
      ```

## Complete Details of All Search Algorithms
1. **Depth-First Search (DFS):**
    - Implementation: `depthFirstSearch` function in `search.py`.
    - Description: DFS explores as far as possible along each branch before backtracking. It uses a stack to keep track of the search frontier.
    - Note: Implemented as graph search to avoid expanding already visited states.

2. **Breadth-First Search (BFS):**
    - Implementation: `breadthFirstSearch` function in `search.py`.
    - Description: BFS explores all neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.
    - Note: Implemented as graph search to avoid expanding already visited states.

3. **Uniform-Cost Search (UCS):**
    - Implementation: `uniformCostSearch` function in `search.py`.
    - Description: UCS expands the node with the lowest path cost, keeping the frontier sorted by path cost.
    - Note: Utilizes priority queue for frontier management.

4. **A* Search:**
    - Implementation: `aStarSearch` function in `search.py`.
    - Description: A* evaluates nodes based on the sum of the cost to reach the node and the estimated cost to the goal. It selects the node with the lowest combined cost.
    - Note: Requires a heuristic function for estimating the cost from a node to the goal.

## Summary
This project involves implementing DFS, BFS, UCS, and A* search algorithms to help Pac-Man navigate through maze environments efficiently. Each algorithm serves a specific purpose, with variations in exploration strategy and frontier management. Additionally, heuristic functions are utilized in A* search to improve pathfinding efficiency.

## Reference
- Textbook: [Add relevant textbook reference here]
- Lecture Slides: [Add relevant lecture slide reference here]

