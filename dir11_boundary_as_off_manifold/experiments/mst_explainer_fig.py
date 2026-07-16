#!/usr/bin/env python3
"""
dir11 — MST explainer schematic (operator feedback 2026-07-16, point 1:
"I don't understand minimum spanning tree. explain It.").

Toy 2D illustration, NOT a result: (a) what a minimum spanning tree is;
(b) what the bottleneck B(u,v) — the largest edge on the MST path — means.
Deterministic (fixed seed). Output: plots/mst_explainer.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, 'plots')

rng = np.random.default_rng(3)
# two loose blobs with a sparse trickle of points between them
A = rng.normal([0.0, 0.0], 0.55, size=(16, 2))
B = rng.normal([4.2, 0.6], 0.55, size=(16, 2))
mid = np.array([[1.55, 0.05], [2.25, 0.35], [3.0, 0.25]]) + rng.normal(0, 0.08, (3, 2))
X = np.vstack([A, mid, B])

D = squareform(pdist(X))
T = minimum_spanning_tree(D).toarray()
T = T + T.T  # symmetrize for path lookup

# unique MST path between u (in A) and v (in B) via predecessors
u = int(np.argmin(np.linalg.norm(A - [-0.6, -0.2], axis=1)))          # a point deep in A
v = len(A) + len(mid) + int(np.argmin(np.linalg.norm(B - [4.9, 0.8], axis=1)))  # deep in B
_, pred = shortest_path(T, directed=False, return_predecessors=True, indices=u)
path = [v]
while path[-1] != u:
    path.append(pred[path[-1]])
path = path[::-1]
edges_on_path = list(zip(path[:-1], path[1:]))
blen = [np.linalg.norm(X[i] - X[j]) for i, j in edges_on_path]
bi, bj = edges_on_path[int(np.argmax(blen))]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, title in zip(axes, [
        '(a) The minimum spanning tree (MST):\nconnect ALL points using the shortest total edge length',
        '(b) Bottleneck B(u,v): the LARGEST single edge\non the unique MST path from u to v']):
    ax.scatter(X[:, 0], X[:, 1], s=28, c='#888888', zorder=3)
    ii, jj = np.nonzero(np.triu(T))
    for i, j in zip(ii, jj):
        ax.plot(*X[[i, j]].T, color='#bbbbbb', lw=1.2, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')

ax = axes[1]
for i, j in edges_on_path:
    ax.plot(*X[[i, j]].T, color='tab:blue', lw=2.2, zorder=2)
ax.plot(*X[[bi, bj]].T, color='tab:red', lw=3.5, zorder=4)
ax.scatter(*X[[u, v]].T, s=90, c='k', zorder=5)
ax.annotate('u', X[u], textcoords='offset points', xytext=(-12, -4), fontsize=12)
ax.annotate('v', X[v], textcoords='offset points', xytext=(8, -4), fontsize=12)
mid_pt = (X[bi] + X[bj]) / 2
ax.annotate('bottleneck B(u,v)\n= the one big hop\nthe traveler cannot avoid',
            mid_pt, textcoords='offset points', xytext=(-30, -58),
            fontsize=9, color='tab:red',
            arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.2))
ax.annotate('small steps inside\na dense region', X[path[1]],
            textcoords='offset points', xytext=(-58, 30), fontsize=9, color='tab:blue',
            arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1.0))

fig.suptitle('Toy 2-D illustration (schematic, not data): stepping through a point cloud', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(PLOTS, 'mst_explainer.png')
fig.savefig(out, dpi=150)
plt.close(fig)
print('saved', out)
