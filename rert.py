import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt


class ReRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start, finish, obstacles, bounds, expand_dis, epoch):
        """
        bounds  : [min, max] for both X and Y axes
        expand_dis : step size per expansion
        epoch   : max iterations
        """
        self.start = self.Node(start[0], start[1])
        self.finish = self.Node(finish[0], finish[1])
        self.bounds = bounds
        self.expand_dis = expand_dis
        self.epoch = epoch
        self.obstacles = obstacles
        self.node_list = []

    # ------------------------------------------------------------------
    def path_planning(self):
        """
        Returns
        -------
        path : list of [x, y] from finish → start, or None if not found
        iterations_used : int  (number of epochs actually run)
        """
        self.node_list = [self.start]

        for i in range(self.epoch):
            rnd_node = self.Node(
                np.random.randint(self.bounds[0], self.bounds[1]),
                np.random.randint(self.bounds[0], self.bounds[1]),
            )

            # nearest node by squared Euclidean distance
            dlist = [
                (node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                for node in self.node_list
            ]
            nearest_node = self.node_list[dlist.index(min(dlist))]

            new_node = self.get_new_node(nearest_node, rnd_node, self.expand_dis)

            if self.check_node_safe(new_node):
                self.node_list.append(new_node)

                a = new_node.x - self.finish.x
                b = new_node.y - self.finish.y
                dist_to_finish = np.hypot(a, b)

                if dist_to_finish < self.expand_dis:
                    final_node = self.get_new_node(new_node, self.finish, self.expand_dis)
                    if self.check_node_safe(final_node):
                        path = [[self.finish.x, self.finish.y]]
                        node = self.node_list[-1]
                        while node.parent is not None:
                            path.append([node.x, node.y])
                            node = node.parent
                        path.append([self.start.x, self.start.y])
                        return path, i + 1

        return None, self.epoch

    # ------------------------------------------------------------------
    def get_new_node(self, from_node, to_node, extend_dis):
        new_node = self.Node(from_node.x, from_node.y)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        a = to_node.x - from_node.x
        b = to_node.y - from_node.y
        c = np.hypot(a, b)

        if extend_dis > c:
            extend_dis = c

        n_expand = int(np.floor(extend_dis))
        t = np.arctan2(b, a)

        for _ in range(n_expand):
            new_node.x += np.cos(t)
            new_node.y += np.sin(t)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        new_node.parent = from_node
        return new_node

    # ------------------------------------------------------------------
    def check_node_safe(self, node):
        for (ox, oy, radius) in self.obstacles:
            dx2 = [(ox - x) ** 2 for x in node.path_x]
            dy2 = [(oy - y) ** 2 for y in node.path_y]
            if min(d + e for d, e in zip(dx2, dy2)) < radius ** 2:
                return False
        return True

    # ------------------------------------------------------------------
    def compute_path_length(self, path):
        """Euclidean length of the returned path."""
        if path is None:
            return float("nan")
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i][0] - path[i + 1][0]
            dy = path[i][1] - path[i + 1][1]
            total += np.hypot(dx, dy)
        return total

    # ------------------------------------------------------------------
    def save_figure(self, path, filepath):
        """Save a static plot of the final state to *filepath*."""
        fig, ax = plt.subplots(figsize=(7, 7))

        # tree edges
        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, "-b", linewidth=0.6, alpha=0.5)

        # obstacles
        deg = list(range(0, 361))
        for (ox, oy, r) in self.obstacles:
            xl = [ox + r * np.cos(np.radians(d)) for d in deg]
            yl = [oy + r * np.sin(np.radians(d)) for d in deg]
            ax.fill(xl, yl, color="salmon", alpha=0.6)
            ax.plot(xl, yl, "-r", linewidth=1)

        # start / finish
        ax.plot(self.start.x, self.start.y, "^b", markersize=10, label="Start")
        ax.plot(self.finish.x, self.finish.y, "^g", markersize=10, label="Finish")

        # solution path
        if path is not None:
            ax.plot(
                [p[0] for p in path],
                [p[1] for p in path],
                "-g",
                linewidth=2.5,
                label="Path",
            )

        lo, hi = self.bounds
        ax.set_xlim(lo - 1, hi + 1)
        ax.set_ylim(lo - 1, hi + 1)
        ax.set_xticks(np.arange(lo, hi + 1))
        ax.set_yticks(np.arange(lo, hi + 1))
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(
            f"epoch={self.epoch}  expand_dis={self.expand_dis}  "
            f"nodes={len(self.node_list)}",
            fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(filepath, dpi=120)
        plt.close(fig)
