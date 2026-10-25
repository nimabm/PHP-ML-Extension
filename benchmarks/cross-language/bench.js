// Usage: node bench.js <data dir> <label>
'use strict';
const fs = require('fs');

function load(path) {
  const lines = fs.readFileSync(path, 'utf8').split('\n').filter((l) => l.trim() !== '');
  lines.shift();
  return lines.map((l) => l.split(' ').map(Number));
}

function bench(label, name, fn, check) {
  const repeats = Number(process.env.REPEATS || 5);
  let result = fn(); // warm-up
  const times = [];
  for (let r = 0; r < repeats; r++) {
    const t = process.hrtime.bigint();
    result = fn();
    times.push(Number(process.hrtime.bigint() - t) / 1e6);
  }
  times.sort((a, b) => a - b);
  console.log(`${label}|${name}|${times[repeats >> 1].toFixed(3)}|${times[0].toFixed(3)}|${check(result)}`);
}

function knnPredict(X, labels, queries, k) {
  const out = [];
  for (const q of queries) {
    const dist = [];
    for (let i = 0; i < X.length; i++) {
      const row = X[i];
      let s = 0.0;
      for (let j = 0; j < row.length; j++) {
        const diff = row[j] - q[j];
        s += diff * diff;
      }
      dist.push([s, i]);
    }
    dist.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const votes = new Map();
    for (let n = 0; n < k; n++) {
      const l = labels[dist[n][1]];
      votes.set(l, (votes.get(l) || 0) + 1);
    }
    let best = null;
    let bestVotes = -1;
    for (const [l, v] of votes) {
      if (v > bestVotes) { best = l; bestVotes = v; }
    }
    out.push(best);
  }
  return out;
}

function logisticFit(X, y, lr, iterations) {
  const d = X[0].length;
  const n = X.length;
  const w = new Array(d).fill(0.0);
  let b = 0.0;
  for (let it = 0; it < iterations; it++) {
    const gw = new Array(d).fill(0.0);
    let gb = 0.0;
    for (let i = 0; i < n; i++) {
      const row = X[i];
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * row[j];
      const err = 1.0 / (1.0 + Math.exp(-z)) - y[i];
      gb += err;
      for (let j = 0; j < d; j++) gw[j] += err * row[j];
    }
    for (let j = 0; j < d; j++) w[j] -= (lr * gw[j]) / n;
    b -= (lr * gb) / n;
  }
  return [w, b];
}

function linearFit(X, y) {
  const d = X[0].length + 1;
  const A = Array.from({ length: d }, () => new Array(d).fill(0.0));
  const rhs = new Array(d).fill(0.0);
  for (let i = 0; i < X.length; i++) {
    const row = X[i].concat([1.0]);
    for (let a = 0; a < d; a++) {
      rhs[a] += row[a] * y[i];
      const Aa = A[a];
      for (let c = 0; c < d; c++) Aa[c] += row[a] * row[c];
    }
  }
  for (let p = 0; p < d; p++) {
    for (let r = p + 1; r < d; r++) {
      const f = A[r][p] / A[p][p];
      for (let c = p; c < d; c++) A[r][c] -= f * A[p][c];
      rhs[r] -= f * rhs[p];
    }
  }
  const w = new Array(d).fill(0.0);
  for (let r = d - 1; r >= 0; r--) {
    let s = rhs[r];
    for (let c = r + 1; c < d; c++) s -= A[r][c] * w[c];
    w[r] = s / A[r][r];
  }
  return w;
}

function kmeans(X, seeds, iterations) {
  const centroids = seeds.map((c) => c.slice());
  const k = centroids.length;
  const d = X[0].length;
  for (let it = 0; it < iterations; it++) {
    const sums = Array.from({ length: k }, () => new Array(d).fill(0.0));
    const counts = new Array(k).fill(0);
    for (const x of X) {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        const center = centroids[c];
        let dist = 0.0;
        for (let j = 0; j < d; j++) {
          const diff = x[j] - center[j];
          dist += diff * diff;
        }
        if (dist < bestDist) { bestDist = dist; best = c; }
      }
      counts[best]++;
      const s = sums[best];
      for (let j = 0; j < d; j++) s[j] += x[j];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) for (let j = 0; j < d; j++) centroids[c][j] = sums[c][j] / counts[c];
    }
  }
  return centroids;
}

// CART with Gini impurity on two classes; nodes are [feature, threshold, left, right, class].
function treeFit(X, y, maxHeight, maxLeaf, minPurity) {
  const d = X[0].length;
  const nodes = [null];
  const stack = [[0, X.map((_, i) => i), 0]];
  while (stack.length) {
    const [id, rows, depth] = stack.pop();
    const n = rows.length;
    let c1 = 0;
    for (const r of rows) c1 += y[r];
    const c0 = n - c1;
    const impurity = n - (c0 * c0 + c1 * c1) / n;
    const leaf = [-1, 0.0, 0, 0, c1 > c0 ? 1 : 0];
    if (depth >= maxHeight || n <= maxLeaf || impurity <= 1e-12) { nodes[id] = leaf; continue; }
    let bestGain = 0.0, bestF = -1, bestT = 0.0;
    for (let f = 0; f < d; f++) {
      const order = rows.slice().sort((a, b) => X[a][f] - X[b][f]);
      const v = order.map((r) => X[r][f]);
      if (!(v[n - 1] > v[0])) continue;
      let l0 = 0, l1 = 0;
      for (let i = 0; i < n - 1; i++) {
        if (y[order[i]]) l1++; else l0++;
        if (v[i] === v[i + 1]) continue;
        const ln = i + 1, rn = n - ln, r0 = c0 - l0, r1 = c1 - l1;
        const gain = impurity - (ln - (l0 * l0 + l1 * l1) / ln) - (rn - (r0 * r0 + r1 * r1) / rn);
        if (gain > bestGain) { bestGain = gain; bestF = f; bestT = v[i] + (v[i + 1] - v[i]) / 2; }
      }
    }
    if (bestF < 0 || bestGain / n < minPurity) { nodes[id] = leaf; continue; }
    const left = [], right = [];
    for (const r of rows) (X[r][bestF] < bestT ? left : right).push(r);
    if (!left.length || !right.length) { nodes[id] = leaf; continue; }
    const l = nodes.length;
    nodes.push(null, null);
    nodes[id] = [bestF, bestT, l, l + 1, 0];
    stack.push([l + 1, right, depth + 1], [l, left, depth + 1]);
  }
  return nodes;
}

function treePredict(nodes, queries) {
  return queries.map((q) => {
    let i = 0;
    while (nodes[i][0] >= 0) i = q[nodes[i][0]] < nodes[i][1] ? nodes[i][2] : nodes[i][3];
    return nodes[i][4];
  });
}

// Dense ReLU layers, linear output; W[l] is out x in. Returns every layer's activations.
function mlpForward(W, b, x) {
  const acts = [x];
  const L = W.length;
  for (let l = 0; l < L; l++) {
    const a = new Array(W[l].length);
    for (let o = 0; o < W[l].length; o++) {
      const row = W[l][o];
      let z = b[l][o];
      for (let i = 0; i < row.length; i++) z += row[i] * x[i];
      a[o] = l < L - 1 && z < 0 ? 0.0 : z;
    }
    acts.push(a);
    x = a;
  }
  return acts;
}

function mlpFit(X, y, W0, perms, batch, lr) {
  const W = W0.map((m) => m.map((r) => r.slice()));
  const L = W.length;
  const zeros2 = () => W.map((m) => m.map((r) => new Array(r.length).fill(0.0)));
  const zeros1 = () => W.map((m) => new Array(m.length).fill(0.0));
  const b = zeros1(), mW = zeros2(), vW = zeros2(), mb = zeros1(), vb = zeros1();
  let t = 0;
  for (const perm of perms) {
    for (let s = 0; s < perm.length; s += batch) {
      const rows = perm.slice(s, s + batch);
      const inv = 1.0 / rows.length;
      const gW = zeros2(), gb = zeros1();
      for (const r of rows) {
        const acts = mlpForward(W, b, X[r]);
        const out = acts[L];
        const mx = Math.max(...out);
        const e = out.map((v) => Math.exp(v - mx));
        const tot = e.reduce((a, v) => a + v, 0);
        let delta = e.map((v, k) => (v / tot - (k === y[r] ? 1.0 : 0.0)) * inv);
        for (let l = L - 1; l >= 0; l--) {
          const prev = acts[l];
          const back = l > 0 ? new Array(prev.length).fill(0.0) : null;
          for (let o = 0; o < delta.length; o++) {
            const dv = delta[o];
            gb[l][o] += dv;
            const row = W[l][o], g = gW[l][o];
            for (let i = 0; i < prev.length; i++) {
              g[i] += dv * prev[i];
              if (back !== null) back[i] += row[i] * dv;
            }
          }
          if (back !== null) delta = back.map((v, i) => (prev[i] > 0 ? v : 0.0));
        }
      }
      t++;
      const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let l = 0; l < L; l++) {
        for (let o = 0; o < W[l].length; o++) {
          const row = W[l][o], gRow = gW[l][o], mRow = mW[l][o], vRow = vW[l][o];
          for (let i = 0; i < row.length; i++) {
            const g = gRow[i];
            const m = (mRow[i] = 0.9 * mRow[i] + 0.1 * g);
            const v = (vRow[i] = 0.999 * vRow[i] + 0.001 * g * g);
            row[i] -= (lr * (m / c1)) / (Math.sqrt(v / c2) + 1e-8);
          }
          const g = gb[l][o];
          const m = (mb[l][o] = 0.9 * mb[l][o] + 0.1 * g);
          const v = (vb[l][o] = 0.999 * vb[l][o] + 0.001 * g * g);
          b[l][o] -= (lr * (m / c1)) / (Math.sqrt(v / c2) + 1e-8);
        }
      }
    }
  }
  return [W, b];
}

function mlpCorrect(X, y, W, b) {
  let n = 0;
  X.forEach((x, i) => {
    const out = mlpForward(W, b, x)[W.length];
    if ((out[1] > out[0] ? 1 : 0) === y[i]) n++;
  });
  return n;
}

function logisticPositives(X, w, b) {
  let n = 0;
  for (const row of X) {
    let z = b;
    for (let j = 0; j < row.length; j++) z += w[j] * row[j];
    if (z >= 0) n++;
  }
  return n;
}

const sum = (a) => a.reduce((s, v) => s + v, 0);
const [dir, label] = process.argv.slice(2);
const X = load(`${dir}/cls_X.txt`);
const labels = load(`${dir}/cls_y.txt`).map((r) => r[0] | 0);
const Q = load(`${dir}/knn_Q.txt`);
const Xl = load(`${dir}/lin_X.txt`);
const yl = load(`${dir}/lin_y.txt`).map((r) => r[0]);
const Xk = load(`${dir}/km_X.txt`);
const seeds = Xk.slice(0, 4);
const Xt = load(`${dir}/tree_X.txt`);
const yt = load(`${dir}/tree_y.txt`).map((r) => r[0] | 0);
const Qt = load(`${dir}/tree_Q.txt`);
const W0 = [1, 2, 3].map((i) => load(`${dir}/mlp_W${i}.txt`));
const perms = load(`${dir}/mlp_perm.txt`).map((r) => r.map((v) => v | 0));

bench(label, 'knn', () => knnPredict(X, labels, Q, 5), sum);
bench(label, 'logreg', () => logisticFit(X, labels, 0.5, 200), (m) => logisticPositives(X, m[0], m[1]));
bench(label, 'linreg', () => linearFit(Xl, yl), (w) => sum(w).toFixed(6));
bench(label, 'kmeans', () => kmeans(Xk, seeds, 20), (cs) => sum(cs.map(sum)).toFixed(6));
bench(label, 'tree', () => treeFit(Xt, yt, 10, 3, 1e-7), (t) => sum(treePredict(t, Qt)));
bench(label, 'mlp', () => mlpFit(X, labels, W0, perms, 64, 0.005), (m) => mlpCorrect(X, labels, m[0], m[1]));
