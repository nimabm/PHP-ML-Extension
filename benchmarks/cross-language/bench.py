# Usage: python bench.py <data dir> <pure|numpy|sklearn> <label>
import math
import os
import sys
import time


def load(path):
    with open(path) as f:
        f.readline()
        return [[float(v) for v in line.split()] for line in f if line.strip()]


def bench(label, name, fn, check):
    repeats = int(os.environ.get("REPEATS", "5"))
    result = fn()  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter_ns()
        result = fn()
        times.append((time.perf_counter_ns() - t) / 1e6)
    times.sort()
    print(f"{label}|{name}|{times[repeats // 2]:.3f}|{times[0]:.3f}|{check(result)}", flush=True)


# ---- pure Python implementations -----------------------------------------------

def knn_predict(X, labels, queries, k):
    out = []
    for q in queries:
        dist = []
        for i, row in enumerate(X):
            s = 0.0
            for j, v in enumerate(row):
                diff = v - q[j]
                s += diff * diff
            dist.append((s, i))
        dist.sort()
        votes = {}
        for _, i in dist[:k]:
            votes[labels[i]] = votes.get(labels[i], 0) + 1
        out.append(max(votes, key=votes.get))
    return out


def logistic_fit(X, y, lr, iterations):
    d = len(X[0])
    n = len(X)
    w = [0.0] * d
    b = 0.0
    for _ in range(iterations):
        gw = [0.0] * d
        gb = 0.0
        for i, row in enumerate(X):
            z = b
            for j, v in enumerate(row):
                z += w[j] * v
            err = 1.0 / (1.0 + math.exp(-z)) - y[i]
            gb += err
            for j, v in enumerate(row):
                gw[j] += err * v
        for j in range(d):
            w[j] -= lr * gw[j] / n
        b -= lr * gb / n
    return w, b


def linear_fit(X, y):
    d = len(X[0]) + 1
    A = [[0.0] * d for _ in range(d)]
    rhs = [0.0] * d
    for i, row in enumerate(X):
        row = row + [1.0]
        for a in range(d):
            rhs[a] += row[a] * y[i]
            Aa = A[a]
            ra = row[a]
            for c in range(d):
                Aa[c] += ra * row[c]
    for p in range(d):
        for r in range(p + 1, d):
            f = A[r][p] / A[p][p]
            for c in range(p, d):
                A[r][c] -= f * A[p][c]
            rhs[r] -= f * rhs[p]
    w = [0.0] * d
    for r in range(d - 1, -1, -1):
        s = rhs[r]
        for c in range(r + 1, d):
            s -= A[r][c] * w[c]
        w[r] = s / A[r][r]
    return w


def kmeans(X, centroids, iterations):
    centroids = [list(c) for c in centroids]
    k = len(centroids)
    d = len(X[0])
    for _ in range(iterations):
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for x in X:
            best = 0
            best_dist = math.inf
            for c, center in enumerate(centroids):
                dist = 0.0
                for j, v in enumerate(x):
                    diff = v - center[j]
                    dist += diff * diff
                if dist < best_dist:
                    best_dist = dist
                    best = c
            counts[best] += 1
            s = sums[best]
            for j, v in enumerate(x):
                s[j] += v
        for c in range(k):
            if counts[c] > 0:
                for j in range(d):
                    centroids[c][j] = sums[c][j] / counts[c]
    return centroids


def tree_fit(X, y, max_height, max_leaf, min_purity):
    """CART with Gini impurity on two classes; nodes are [feature, threshold, left, right, class]."""
    d = len(X[0])
    nodes = [None]
    stack = [(0, list(range(len(X))), 0)]
    while stack:
        node_id, rows, depth = stack.pop()
        n = len(rows)
        c1 = sum(y[r] for r in rows)
        c0 = n - c1
        impurity = n - (c0 * c0 + c1 * c1) / n
        leaf = [-1, 0.0, 0, 0, 1 if c1 > c0 else 0]
        if depth >= max_height or n <= max_leaf or impurity <= 1e-12:
            nodes[node_id] = leaf
            continue
        best_gain, best_f, best_t = 0.0, -1, 0.0
        for f in range(d):
            order = sorted(rows, key=lambda r: X[r][f])
            v = [X[r][f] for r in order]
            if not v[-1] > v[0]:
                continue
            l0 = l1 = 0
            for i in range(n - 1):
                if y[order[i]]:
                    l1 += 1
                else:
                    l0 += 1
                if v[i] == v[i + 1]:
                    continue
                ln = i + 1
                rn = n - ln
                r0 = c0 - l0
                r1 = c1 - l1
                gain = impurity - (ln - (l0 * l0 + l1 * l1) / ln) - (rn - (r0 * r0 + r1 * r1) / rn)
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, v[i] + (v[i + 1] - v[i]) / 2
        if best_f < 0 or best_gain / n < min_purity:
            nodes[node_id] = leaf
            continue
        left = [r for r in rows if X[r][best_f] < best_t]
        right = [r for r in rows if not X[r][best_f] < best_t]
        if not left or not right:
            nodes[node_id] = leaf
            continue
        l = len(nodes)
        nodes += [None, None]
        nodes[node_id] = [best_f, best_t, l, l + 1, 0]
        stack.append((l + 1, right, depth + 1))
        stack.append((l, left, depth + 1))
    return nodes


def tree_predict(nodes, queries):
    out = []
    for q in queries:
        i = 0
        while nodes[i][0] >= 0:
            i = nodes[i][2] if q[nodes[i][0]] < nodes[i][1] else nodes[i][3]
        out.append(nodes[i][4])
    return out


def mlp_forward(W, b, x):
    """Dense ReLU layers, linear output; W[l] is out x in. Returns every layer's activations."""
    acts = [x]
    L = len(W)
    for l in range(L):
        a = []
        for o, row in enumerate(W[l]):
            z = b[l][o]
            for i, w in enumerate(row):
                z += w * x[i]
            a.append(0.0 if l < L - 1 and z < 0 else z)
        acts.append(a)
        x = a
    return acts


def mlp_fit(X, y, W, perms, batch, lr):
    W = [[list(r) for r in m] for m in W]
    L = len(W)
    b = [[0.0] * len(m) for m in W]
    mW = [[[0.0] * len(r) for r in m] for m in W]
    vW = [[[0.0] * len(r) for r in m] for m in W]
    mb = [[0.0] * len(m) for m in W]
    vb = [[0.0] * len(m) for m in W]
    t = 0
    for perm in perms:
        for s in range(0, len(perm), batch):
            rows = perm[s:s + batch]
            inv = 1.0 / len(rows)
            gW = [[[0.0] * len(r) for r in m] for m in W]
            gb = [[0.0] * len(m) for m in W]
            for r in rows:
                acts = mlp_forward(W, b, X[r])
                out = acts[L]
                mx = max(out)
                e = [math.exp(v - mx) for v in out]
                tot = sum(e)
                delta = [(v / tot - (1.0 if k == y[r] else 0.0)) * inv for k, v in enumerate(e)]
                for l in range(L - 1, -1, -1):
                    prev = acts[l]
                    back = [0.0] * len(prev) if l > 0 else None
                    for o, dv in enumerate(delta):
                        gb[l][o] += dv
                        row = W[l][o]
                        g = gW[l][o]
                        for i, p in enumerate(prev):
                            g[i] += dv * p
                            if back is not None:
                                back[i] += row[i] * dv
                    if back is not None:
                        delta = [v if prev[i] > 0 else 0.0 for i, v in enumerate(back)]
            t += 1
            c1 = 1 - 0.9 ** t
            c2 = 1 - 0.999 ** t
            for l in range(L):
                for o, row in enumerate(W[l]):
                    g_row, m_row, v_row = gW[l][o], mW[l][o], vW[l][o]
                    for i in range(len(row)):
                        g = g_row[i]
                        m = m_row[i] = 0.9 * m_row[i] + 0.1 * g
                        v = v_row[i] = 0.999 * v_row[i] + 0.001 * g * g
                        row[i] -= lr * (m / c1) / (math.sqrt(v / c2) + 1e-8)
                    g = gb[l][o]
                    m = mb[l][o] = 0.9 * mb[l][o] + 0.1 * g
                    v = vb[l][o] = 0.999 * vb[l][o] + 0.001 * g * g
                    b[l][o] -= lr * (m / c1) / (math.sqrt(v / c2) + 1e-8)
    return W, b


def mlp_correct(X, y, W, b):
    n = 0
    for x, label in zip(X, y):
        out = mlp_forward(W, b, x)[len(W)]
        n += (1 if out[1] > out[0] else 0) == label
    return n


def logistic_positives(X, w, b):
    n = 0
    for row in X:
        z = b
        for j, v in enumerate(row):
            z += w[j] * v
        n += 1 if z >= 0 else 0
    return n


# ---- runs -------------------------------------------------------------------------

data, mode, label = sys.argv[1:4]
X = load(f"{data}/cls_X.txt")
labels = [int(r[0]) for r in load(f"{data}/cls_y.txt")]
Q = load(f"{data}/knn_Q.txt")
Xl = load(f"{data}/lin_X.txt")
yl = [r[0] for r in load(f"{data}/lin_y.txt")]
Xk = load(f"{data}/km_X.txt")
seeds = Xk[:4]
Xt = load(f"{data}/tree_X.txt")
yt = [int(r[0]) for r in load(f"{data}/tree_y.txt")]
Qt = load(f"{data}/tree_Q.txt")
W0 = [load(f"{data}/mlp_W{i}.txt") for i in (1, 2, 3)]
perms = [[int(v) for v in r] for r in load(f"{data}/mlp_perm.txt")]

if mode == "pure":
    bench(label, "knn", lambda: knn_predict(X, labels, Q, 5), sum)
    bench(label, "logreg", lambda: logistic_fit(X, labels, 0.5, 200), lambda m: logistic_positives(X, *m))
    bench(label, "linreg", lambda: linear_fit(Xl, yl), lambda w: f"{sum(w):.6f}")
    bench(label, "kmeans", lambda: kmeans(Xk, seeds, 20), lambda cs: f"{sum(map(sum, cs)):.6f}")
    bench(label, "tree", lambda: tree_fit(Xt, yt, 10, 3, 1e-7), lambda t: sum(tree_predict(t, Qt)))
    bench(label, "mlp", lambda: mlp_fit(X, labels, W0, perms, 64, 0.005), lambda m: mlp_correct(X, labels, *m))
    sys.exit()

import numpy as np

nX, ny, nQ = np.array(X), np.array(labels, dtype=float), np.array(Q)
nXl, nyl, nXk = np.array(Xl), np.array(yl), np.array(Xk)
nseeds = nXk[:4].copy()
iy = np.array(labels)
nXt, nyt, nQt = np.array(Xt), np.array(yt), np.array(Qt)
nW = [np.array(w) for w in W0]
nperms = np.array(perms)

if mode == "numpy":
    # Same algorithms, vectorised with NumPy (BLAS underneath).
    def np_knn():
        d2 = (nQ * nQ).sum(1)[:, None] - 2 * nQ @ nX.T + (nX * nX).sum(1)[None, :]
        idx = np.argpartition(d2, 5, axis=1)[:, :5]
        return (ny[idx].sum(1) >= 3).astype(int)

    def np_logreg():
        w = np.zeros(nX.shape[1])
        b = 0.0
        n = len(nX)
        for _ in range(200):
            err = 1.0 / (1.0 + np.exp(-(nX @ w + b))) - ny
            w -= 0.5 * (nX.T @ err) / n
            b -= 0.5 * err.sum() / n
        return w, b

    def np_linreg():
        A = np.hstack([nXl, np.ones((len(nXl), 1))])
        return np.linalg.solve(A.T @ A, A.T @ nyl)

    def np_kmeans():
        c = nseeds.copy()
        for _ in range(20):
            d2 = ((nXk[:, None, :] - c[None, :, :]) ** 2).sum(2)
            a = d2.argmin(1)
            for j in range(len(c)):
                m = a == j
                if m.any():
                    c[j] = nXk[m].mean(0)
        return c

    def np_mlp_forward(W, b, x):
        acts = [x]
        for l in range(len(W)):
            z = acts[-1] @ W[l].T + b[l]
            acts.append(np.maximum(z, 0.0) if l < len(W) - 1 else z)
        return acts

    def np_mlp():
        W = [w.copy() for w in nW]
        b = [np.zeros(len(w)) for w in W]
        params = W + b
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        onehot = np.eye(2)[iy]
        t = 0
        for perm in nperms:
            for s in range(0, len(perm), 64):
                rows = perm[s:s + 64]
                acts = np_mlp_forward(W, b, nX[rows])
                e = np.exp(acts[-1] - acts[-1].max(1, keepdims=True))
                delta = (e / e.sum(1, keepdims=True) - onehot[rows]) / len(rows)
                gW, gb = [None] * len(W), [None] * len(W)
                for l in range(len(W) - 1, -1, -1):
                    gW[l] = delta.T @ acts[l]
                    gb[l] = delta.sum(0)
                    if l > 0:
                        delta = (delta @ W[l]) * (acts[l] > 0)
                t += 1
                c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
                for p, g, a, v in zip(params, gW + gb, m1, m2):
                    a *= 0.9
                    a += 0.1 * g
                    v *= 0.999
                    v += 0.001 * g * g
                    p -= 0.005 * (a / c1) / (np.sqrt(v / c2) + 1e-8)
        return W, b

    bench(label, "knn", np_knn, lambda p: int(p.sum()))
    bench(label, "logreg", np_logreg, lambda m: int(((nX @ m[0] + m[1]) >= 0).sum()))
    bench(label, "linreg", np_linreg, lambda w: f"{w.sum():.6f}")
    bench(label, "kmeans", np_kmeans, lambda c: f"{c.sum():.6f}")
    bench(label, "mlp", np_mlp, lambda m: int(((lambda o: o[:, 1] > o[:, 0])(np_mlp_forward(*m, nX)[-1]) == iy).sum()))

elif mode == "sklearn":
    import warnings

    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    warnings.filterwarnings("ignore")
    bench(label, "knn", lambda: KNeighborsClassifier(5, algorithm="brute").fit(nX, iy).predict(nQ), lambda p: int(p.sum()))
    # lbfgs to convergence: different optimiser, shown for reference only.
    bench(label, "logreg", lambda: LogisticRegression(C=np.inf, max_iter=200).fit(nX, iy), lambda m: int(m.predict(nX).sum()))
    bench(label, "linreg", lambda: LinearRegression().fit(nXl, nyl), lambda m: f"{m.coef_.sum() + m.intercept_:.6f}")
    bench(label, "kmeans", lambda: KMeans(4, init=nseeds, n_init=1, max_iter=20, tol=0.0, algorithm="lloyd").fit(nXk),
          lambda m: f"{m.cluster_centers_.sum():.6f}")
    bench(label, "tree", lambda: DecisionTreeClassifier(max_depth=10, min_samples_split=4).fit(nXt, nyt),
          lambda m: int(m.predict(nQt).sum()))
    bench(label, "mlp", lambda: MLPClassifier((32, 16), solver="adam", learning_rate_init=0.005, batch_size=64, max_iter=10,
                                              alpha=0.0, tol=0.0, n_iter_no_change=1000, random_state=1).fit(nX, iy),
          lambda m: int((m.predict(nX) == iy).sum()))

elif mode == "torch":
    import torch

    torch.set_num_threads(1)
    tX, ty, tQ = torch.tensor(nX), torch.tensor(ny), torch.tensor(nQ)
    tXl, tyl, tXk = torch.tensor(nXl), torch.tensor(nyl), torch.tensor(nXk)
    tseeds, tiy = torch.tensor(nseeds), torch.tensor(iy)

    def t_knn():
        idx = torch.cdist(tQ, tX).topk(5, largest=False).indices
        return (ty[idx].sum(1) >= 3).int()

    def t_logreg():
        w = torch.zeros(tX.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.SGD([w, b], lr=0.5)
        for _ in range(200):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(tX @ w + b, ty).backward()
            opt.step()
        return w.detach(), b.detach()

    def t_linreg():
        A = torch.cat([tXl, torch.ones(len(tXl), 1, dtype=torch.float64)], 1)
        return torch.linalg.solve(A.T @ A, A.T @ tyl)

    def t_kmeans():
        c = tseeds.clone()
        for _ in range(20):
            a = torch.cdist(tXk, c).argmin(1)
            for j in range(len(c)):
                m = a == j
                if m.any():
                    c[j] = tXk[m].mean(0)
        return c

    def t_mlp():
        net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 2)).double()
        with torch.no_grad():
            for layer, w in zip([net[0], net[2], net[4]], nW):
                layer.weight.copy_(torch.tensor(w))
                layer.bias.zero_()
        opt = torch.optim.Adam(net.parameters(), lr=0.005)
        for perm in nperms:
            for s in range(0, len(perm), 64):
                rows = torch.from_numpy(perm[s:s + 64])
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(tX[rows]), tiy[rows]).backward()
                opt.step()
        return net

    bench(label, "knn", t_knn, lambda p: int(p.sum()))
    bench(label, "logreg", t_logreg, lambda m: int(((tX @ m[0] + m[1]) >= 0).sum()))
    bench(label, "linreg", t_linreg, lambda w: f"{w.sum().item():.6f}")
    bench(label, "kmeans", t_kmeans, lambda c: f"{c.sum().item():.6f}")
    def t_mlp_correct(net):
        with torch.no_grad():
            out = net(tX)
        return int(((out[:, 1] > out[:, 0]).long() == tiy).sum())

    bench(label, "mlp", t_mlp, t_mlp_correct)
