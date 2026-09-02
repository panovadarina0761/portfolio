import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import heapq
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Graph data

POSITIONS = {
    0: (300, 50),
    1: (150, 100),
    2: (300, 100),
    3: (450, 100),
    4: (100, 200),
    5: (200, 200),
    6: (400, 200),
    7: (500, 200),
    8: (75, 300),
    9: (150, 300),
    10: (225, 300),
}

# Directed graph for BFS/DFS
UNWEIGHTED_ADJ = {
    0: [1, 2, 3],
    1: [4, 5],
    2: [5],
    3: [6, 7],
    4: [8, 9],
    5: [10],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
}

# Undirected weighted graph
WEIGHTED_EDGES = [
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 1),
    (1, 4, 4),
    (1, 5, 1),
    (2, 5, 3),
    (3, 6, 2),
    (3, 7, 4),
    (4, 8, 2),
    (4, 9, 3),
    (5, 10, 1),
]

# Maximum-flow network data
FLOW_EDGES = [
    (0, 1, 10),
    (0, 2, 10),
    (0, 3, 10),
    (1, 4, 4),
    (1, 5, 8),
    (2, 5, 9),
    (3, 6, 14),
    (3, 7, 7),
    (4, 8, 10),
    (4, 9, 10),
    (5, 10, 10),
]


# Helper functions
def build_weighted_adj():
    adj = {v: [] for v in POSITIONS}
    for u, v, w in WEIGHTED_EDGES:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


WEIGHTED_ADJ = build_weighted_adj()


def heuristic(a, b):
    """Euclidean distance between two vertices."""
    x1, y1 = POSITIONS[a]
    x2, y2 = POSITIONS[b]
    return math.hypot(x1 - x2, y1 - y2)


def reconstruct_path(parent, start, goal):
    if goal not in parent and goal != start:
        return []

    path = [goal]
    current = goal

    while current != start:
        current = parent[current]
        path.append(current)

    path.reverse()
    return path



# Algorithms implemented as generators
# Each yield returns the current state for animation

def bfs_steps(start=0):
    visited = {start}
    q = deque([start])
    order = []

    while q:
        current = q.popleft()
        order.append(current)

        yield {
            "visited": set(order),
            "current": current,
            "result": f"BFS: {' -> '.join(map(str, order))}"
        }

        for neighbor in UNWEIGHTED_ADJ[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)


def dfs_steps(start=0):
    visited = set()
    stack = [start]
    order = []

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        order.append(current)

        yield {
            "visited": set(order),
            "current": current,
            "result": f"DFS: {' -> '.join(map(str, order))}"
        }

        # Preserve the traversal order used in the original C++ implementation
        # by sorting neighbors by x-coordinate in descending order.
        neighbors = sorted(
            UNWEIGHTED_ADJ[current],
            key=lambda v: POSITIONS[v][0],
            reverse=True
        )

        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)


def prim_steps(start=0):
    n = len(POSITIONS)
    in_mst = set()
    min_weight = {v: math.inf for v in POSITIONS}
    parent = {v: None for v in POSITIONS}
    min_weight[start] = 0

    total_weight = 0

    while len(in_mst) < n:
        candidates = [v for v in POSITIONS if v not in in_mst]
        if not candidates:
            break

        u = min(candidates, key=lambda v: min_weight[v])

        if min_weight[u] == math.inf:
            break

        in_mst.add(u)

        if parent[u] is not None:
            total_weight += min_weight[u]

        for v, w in WEIGHTED_ADJ[u]:
            if v not in in_mst and w < min_weight[v]:
                min_weight[v] = w
                parent[v] = u

        mst_edges = []
        for v in in_mst:
            if parent[v] is not None:
                mst_edges.append((parent[v], v))

        yield {
            "visited": set(in_mst),
            "current": u,
            "highlight_edges": mst_edges,
            "result": f"MST total weight: {total_weight:g}"
        }


def dijkstra_steps(start=0, goal=10):
    dist = {v: math.inf for v in POSITIONS}
    parent = {}
    visited = set()
    pq = [(0, start)]
    dist[start] = 0

    while pq:
        current_dist, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)

        path = []
        if u == goal:
            path = reconstruct_path(parent, start, goal)

        yield {
            "visited": set(visited),
            "current": u,
            "path": path,
            "result": (
                f"Shortest path cost {start} → {goal}: {dist[goal]:g}"
                if path else
                f"Current vertex: {u}, distance[{u}] = {dist[u]:g}"
            )
        }

        if u == goal:
            break

        for v, w in WEIGHTED_ADJ[u]:
            new_dist = current_dist + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))


def astar_steps(start=0, goal=10):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    closed = set()
    open_nodes = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        open_nodes.discard(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            yield {
                "visited": set(closed) | {current},
                "current": current,
                "open_nodes": set(open_nodes),
                "path": path,
                "result": f"A*: path {' -> '.join(map(str, path))}, cost = {g_score[goal]:g}"
            }
            return

        closed.add(current)

        for neighbor, weight in WEIGHTED_ADJ[current]:
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + weight

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                open_nodes.add(neighbor)

        yield {
            "visited": set(closed),
            "current": current,
            "open_nodes": set(open_nodes),
            "result": (
                f"A*: current={current}, "
                f"g={g_score[current]:.2f}, f={f_score[current]:.2f}"
            )
        }


def ford_fulkerson_steps(start=0, goal=10):
    # residual[u][v] stores the residual capacity
    residual = {u: {} for u in POSITIONS}
    original = {}

    for u, v, capacity in FLOW_EDGES:
        residual[u][v] = capacity
        residual[v].setdefault(u, 0)
        original[(u, v)] = capacity

    max_flow = 0

    while True:
        parent = {start: None}
        q = deque([start])

        while q and goal not in parent:
            u = q.popleft()

            for v, capacity in residual[u].items():
                if capacity > 0 and v not in parent:
                    parent[v] = u
                    q.append(v)

                    if v == goal:
                        break

        if goal not in parent:
            break

        path_flow = math.inf
        v = goal

        while v != start:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        path = []
        v = goal
        while v is not None:
            path.append(v)
            v = parent[v]
        path.reverse()

        v = goal
        while v != start:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] = residual[v].get(u, 0) + path_flow
            v = u

        max_flow += path_flow

        flow_labels = {}
        for (u, v), capacity in original.items():
            current_flow = capacity - residual[u][v]
            flow_labels[(u, v)] = f"{current_flow:g}/{capacity:g}"

        yield {
            "visited": set(path),
            "current": goal,
            "path": path,
            "flow_labels": flow_labels,
            "result": (
                f"Augmenting path: {' -> '.join(map(str, path))}; "
                f"+{path_flow:g}; max flow = {max_flow:g}"
            )
        }

    flow_labels = {}
    for (u, v), capacity in original.items():
        current_flow = capacity - residual[u][v]
        flow_labels[(u, v)] = f"{current_flow:g}/{capacity:g}"

    yield {
        "visited": set(),
        "current": None,
        "flow_labels": flow_labels,
        "result": f"Maximum flow = {max_flow:g}"
    }



# GUI

class GraphAlgorithmsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Algorithm Visualizer")
        self.root.geometry("1180x760")

        self.algorithm = tk.StringVar(value="BFS")
        self.start_vertex = tk.IntVar(value=0)
        self.goal_vertex = tk.IntVar(value=10)
        self.speed_ms = tk.IntVar(value=700)

        self.generator = None
        self.auto_running = False
        self.last_state = {}

        self.create_widgets()
        self.reset_visualization()

    def create_widgets(self):
        control = ttk.Frame(self.root, padding=10)
        control.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            control,
            text="Graph Algorithm Visualizer",
            font=("Segoe UI", 17, "bold")
        ).pack(pady=(0, 4))

        ttk.Label(
            control,
            text="Interactive exploration of classic graph algorithms",
            wraplength=230,
            justify=tk.LEFT
        ).pack(pady=(0, 16))

        ttk.Label(control, text="Algorithm").pack(anchor="w")

        algorithm_box = ttk.Combobox(
            control,
            textvariable=self.algorithm,
            values=[
                "BFS",
                "DFS",
                "Prim",
                "Dijkstra",
                "A*",
                "Ford-Fulkerson"
            ],
            state="readonly",
            width=20
        )
        algorithm_box.pack(fill=tk.X, pady=(3, 12))
        algorithm_box.bind("<<ComboboxSelected>>", lambda event: self.reset_visualization())

        ttk.Label(control, text="Start vertex").pack(anchor="w")

        ttk.Spinbox(
            control,
            from_=0,
            to=10,
            textvariable=self.start_vertex,
            width=8
        ).pack(anchor="w", pady=(3, 10))

        ttk.Label(control, text="Target vertex").pack(anchor="w")

        ttk.Spinbox(
            control,
            from_=0,
            to=10,
            textvariable=self.goal_vertex,
            width=8
        ).pack(anchor="w", pady=(3, 10))

        ttk.Label(control, text="Animation speed (ms)").pack(anchor="w")

        ttk.Scale(
            control,
            from_=150,
            to=1600,
            variable=self.speed_ms,
            orient=tk.HORIZONTAL
        ).pack(fill=tk.X, pady=(3, 15))

        ttk.Button(
            control,
            text="Initialize",
            command=self.start_algorithm
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            control,
            text="Step",
            command=self.next_step
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            control,
            text="Run",
            command=self.auto_run
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            control,
            text="Pause",
            command=self.pause
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            control,
            text="Reset",
            command=self.reset_visualization
        ).pack(fill=tk.X, pady=4)

        ttk.Separator(control, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        ttk.Label(
            control,
            text="Execution status",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")

        self.result_label = ttk.Label(
            control,
            text="Select an algorithm and initialize the simulation.",
            wraplength=230,
            justify=tk.LEFT
        )
        self.result_label.pack(fill=tk.X, pady=8)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_generator(self):
        start = self.start_vertex.get()
        goal = self.goal_vertex.get()
        alg = self.algorithm.get()

        if start not in POSITIONS or goal not in POSITIONS:
            messagebox.showerror("Invalid input", "Vertex IDs must be between 0 and 10.")
            return None

        if alg == "BFS":
            return bfs_steps(start)

        if alg == "DFS":
            return dfs_steps(start)

        if alg == "Prim":
            return prim_steps(start)

        if alg == "Dijkstra":
            return dijkstra_steps(start, goal)

        if alg == "A*":
            return astar_steps(start, goal)

        if alg == "Ford-Fulkerson":
            return ford_fulkerson_steps(start, goal)

        return None

    def start_algorithm(self):
        self.auto_running = False
        self.generator = self.create_generator()

        if self.generator is None:
            return

        self.last_state = {}
        self.result_label.config(text=f"Initialized: {self.algorithm.get()}")
        self.draw_graph({})

    def next_step(self):
        if self.generator is None:
            self.start_algorithm()

        if self.generator is None:
            return

        try:
            state = next(self.generator)
            self.last_state = state
            self.result_label.config(text=state.get("result", ""))
            self.draw_graph(state)

        except StopIteration:
            self.result_label.config(
                text=self.result_label.cget("text") + "\nExecution completed."
            )
            self.generator = None
            self.auto_running = False

    def auto_run(self):
        if self.generator is None:
            self.start_algorithm()

        if self.generator is None:
            return

        self.auto_running = True
        self.run_auto_step()

    def run_auto_step(self):
        if not self.auto_running:
            return

        if self.generator is None:
            self.auto_running = False
            return

        try:
            state = next(self.generator)
            self.last_state = state
            self.result_label.config(text=state.get("result", ""))
            self.draw_graph(state)

            self.root.after(int(self.speed_ms.get()), self.run_auto_step)

        except StopIteration:
            self.auto_running = False
            self.generator = None
            self.result_label.config(
                text=self.result_label.cget("text") + "\nExecution completed."
            )

    def pause(self):
        self.auto_running = False

    def reset_visualization(self):
        self.auto_running = False
        self.generator = None
        self.last_state = {}

        if hasattr(self, "result_label"):
            self.result_label.config(
                text="Select an algorithm and initialize the simulation."
            )

        if hasattr(self, "ax"):
            self.draw_graph({})

    def get_base_edges(self):
        alg = self.algorithm.get()

        if alg in ("BFS", "DFS"):
            edges = []
            for u, neighbors in UNWEIGHTED_ADJ.items():
                for v in neighbors:
                    edges.append((u, v, None))
            return edges

        if alg == "Ford-Fulkerson":
            return [(u, v, capacity) for u, v, capacity in FLOW_EDGES]

        return [(u, v, w) for u, v, w in WEIGHTED_EDGES]

    def draw_graph(self, state):
        self.ax.clear()

        visited = state.get("visited", set())
        current = state.get("current")
        open_nodes = state.get("open_nodes", set())
        path = state.get("path", [])
        highlighted_edges = state.get("highlight_edges", [])
        flow_labels = state.get("flow_labels", {})

        path_edges = set()
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            path_edges.add((a, b))
            path_edges.add((b, a))

        mst_edges = set()
        for a, b in highlighted_edges:
            mst_edges.add((a, b))
            mst_edges.add((b, a))

        # Edges
        for u, v, value in self.get_base_edges():
            x1, y1 = POSITIONS[u]
            x2, y2 = POSITIONS[v]

            linewidth = 1.8

            if (u, v) in path_edges or (u, v) in mst_edges:
                linewidth = 4.2

            self.ax.plot([x1, x2], [y1, y2], linewidth=linewidth)

            if self.algorithm.get() in ("Prim", "Dijkstra", "A*") and value is not None:
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2

                self.ax.text(
                    mx,
                    my,
                    str(value),
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
                )

            if self.algorithm.get() == "Ford-Fulkerson":
                label = flow_labels.get((u, v), f"0/{value}")

                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2

                self.ax.text(
                    mx,
                    my,
                    label,
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
                )

                # Direction arrow
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)

                if length > 0:
                    ux = dx / length
                    uy = dy / length

                    sx = x1 + 24 * ux
                    sy = y1 + 24 * uy
                    ex = x2 - 24 * ux
                    ey = y2 - 24 * uy

                    self.ax.annotate(
                        "",
                        xy=(ex, ey),
                        xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", lw=1.2)
                    )

        # Vertices
        for vertex, (x, y) in POSITIONS.items():
            size = 900

            if vertex == current:
                size = 1200
            elif vertex in open_nodes:
                size = 1050
            elif vertex in visited:
                size = 1000

            self.ax.scatter(x, y, s=size, edgecolors="black", zorder=3)

            self.ax.text(
                x,
                y,
                str(vertex),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                zorder=4
            )

        self.ax.set_title(f"{self.algorithm.get()} — execution trace")
        self.ax.set_xlim(20, 560)
        self.ax.set_ylim(350, 0)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.canvas.draw()


def main():
    root = tk.Tk()
    GraphAlgorithmsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
