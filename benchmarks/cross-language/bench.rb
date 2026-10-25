# Usage: ruby bench.rb <data dir> <label>

def load(path)
  lines = File.readlines(path, chomp: true).reject { |l| l.strip.empty? }
  lines.shift
  lines.map { |l| l.split(' ').map(&:to_f) }
end

def bench(label, name, check)
  repeats = (ENV['REPEATS'] || '5').to_i
  result = yield # warm-up
  times = []
  repeats.times do
    t = Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)
    result = yield
    times << (Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond) - t) / 1e6
  end
  times.sort!
  puts format('%s|%s|%.3f|%.3f|%s', label, name, times[repeats / 2], times[0], check.call(result))
  $stdout.flush
end

def knn_predict(x, labels, queries, k)
  queries.map do |q|
    dist = []
    x.each_with_index do |row, i|
      s = 0.0
      row.each_with_index do |v, j|
        diff = v - q[j]
        s += diff * diff
      end
      dist << [s, i]
    end
    dist.sort!
    votes = Hash.new(0)
    dist.first(k).each { |_, i| votes[labels[i]] += 1 }
    votes.max_by { |_, v| v }[0]
  end
end

def logistic_fit(x, y, lr, iterations)
  d = x[0].size
  n = x.size
  w = Array.new(d, 0.0)
  b = 0.0
  iterations.times do
    gw = Array.new(d, 0.0)
    gb = 0.0
    x.each_with_index do |row, i|
      z = b
      row.each_with_index { |v, j| z += w[j] * v }
      err = 1.0 / (1.0 + Math.exp(-z)) - y[i]
      gb += err
      row.each_with_index { |v, j| gw[j] += err * v }
    end
    d.times { |j| w[j] -= lr * gw[j] / n }
    b -= lr * gb / n
  end
  [w, b]
end

def linear_fit(x, y)
  d = x[0].size + 1
  a = Array.new(d) { Array.new(d, 0.0) }
  rhs = Array.new(d, 0.0)
  x.each_with_index do |r, i|
    row = r + [1.0]
    d.times do |p|
      rhs[p] += row[p] * y[i]
      ap = a[p]
      rp = row[p]
      d.times { |c| ap[c] += rp * row[c] }
    end
  end
  d.times do |p|
    (p + 1...d).each do |r|
      f = a[r][p] / a[p][p]
      (p...d).each { |c| a[r][c] -= f * a[p][c] }
      rhs[r] -= f * rhs[p]
    end
  end
  w = Array.new(d, 0.0)
  (d - 1).downto(0) do |r|
    s = rhs[r]
    (r + 1...d).each { |c| s -= a[r][c] * w[c] }
    w[r] = s / a[r][r]
  end
  w
end

def kmeans(x, seeds, iterations)
  centroids = seeds.map(&:dup)
  k = centroids.size
  d = x[0].size
  iterations.times do
    sums = Array.new(k) { Array.new(d, 0.0) }
    counts = Array.new(k, 0)
    x.each do |pt|
      best = 0
      best_dist = Float::INFINITY
      centroids.each_with_index do |center, c|
        dist = 0.0
        pt.each_with_index do |v, j|
          diff = v - center[j]
          dist += diff * diff
        end
        if dist < best_dist
          best_dist = dist
          best = c
        end
      end
      counts[best] += 1
      s = sums[best]
      pt.each_with_index { |v, j| s[j] += v }
    end
    k.times do |c|
      next unless counts[c] > 0
      d.times { |j| centroids[c][j] = sums[c][j] / counts[c] }
    end
  end
  centroids
end

# CART with Gini impurity on two classes; nodes are [feature, threshold, left, right, class].
def tree_fit(x, y, max_height, max_leaf, min_purity)
  d = x[0].size
  nodes = [nil]
  stack = [[0, (0...x.size).to_a, 0]]
  until stack.empty?
    id, rows, depth = stack.pop
    n = rows.size
    c1 = rows.sum { |r| y[r] }
    c0 = n - c1
    impurity = n - (c0 * c0 + c1 * c1).to_f / n
    leaf = [-1, 0.0, 0, 0, c1 > c0 ? 1 : 0]
    if depth >= max_height || n <= max_leaf || impurity <= 1e-12
      nodes[id] = leaf
      next
    end
    best_gain = 0.0
    best_f = -1
    best_t = 0.0
    d.times do |f|
      order = rows.sort_by { |r| x[r][f] }
      v = order.map { |r| x[r][f] }
      next unless v[-1] > v[0]
      l0 = l1 = 0
      (n - 1).times do |i|
        if y[order[i]] == 1 then l1 += 1 else l0 += 1 end
        next if v[i] == v[i + 1]
        ln = i + 1
        rn = n - ln
        r0 = c0 - l0
        r1 = c1 - l1
        gain = impurity - (ln - (l0 * l0 + l1 * l1).to_f / ln) - (rn - (r0 * r0 + r1 * r1).to_f / rn)
        if gain > best_gain
          best_gain = gain
          best_f = f
          best_t = v[i] + (v[i + 1] - v[i]) / 2
        end
      end
    end
    if best_f < 0 || best_gain / n < min_purity
      nodes[id] = leaf
      next
    end
    left, right = rows.partition { |r| x[r][best_f] < best_t }
    if left.empty? || right.empty?
      nodes[id] = leaf
      next
    end
    l = nodes.size
    nodes.push(nil, nil)
    nodes[id] = [best_f, best_t, l, l + 1, 0]
    stack.push([l + 1, right, depth + 1], [l, left, depth + 1])
  end
  nodes
end

def tree_predict(nodes, queries)
  queries.map do |q|
    i = 0
    i = q[nodes[i][0]] < nodes[i][1] ? nodes[i][2] : nodes[i][3] while nodes[i][0] >= 0
    nodes[i][4]
  end
end

# Dense ReLU layers, linear output; w[l] is out x in. Returns every layer's activations.
def mlp_forward(w, b, x)
  acts = [x]
  last = w.size - 1
  w.each_with_index do |layer, l|
    a = layer.each_with_index.map do |row, o|
      z = b[l][o]
      row.each_with_index { |wv, i| z += wv * x[i] }
      l < last && z < 0 ? 0.0 : z
    end
    acts << a
    x = a
  end
  acts
end

def mlp_fit(x, y, w0, perms, batch, lr)
  w = w0.map { |m| m.map(&:dup) }
  nl = w.size
  zeros2 = -> { w.map { |m| m.map { |r| Array.new(r.size, 0.0) } } }
  zeros1 = -> { w.map { |m| Array.new(m.size, 0.0) } }
  b = zeros1.call
  mw = zeros2.call
  vw = zeros2.call
  mb = zeros1.call
  vb = zeros1.call
  t = 0
  perms.each do |perm|
    perm.each_slice(batch) do |rows|
      inv = 1.0 / rows.size
      gw = zeros2.call
      gb = zeros1.call
      rows.each do |r|
        acts = mlp_forward(w, b, x[r])
        out = acts[nl]
        mx = out.max
        e = out.map { |v| Math.exp(v - mx) }
        tot = e.sum
        delta = e.each_with_index.map { |v, k| (v / tot - (k == y[r] ? 1.0 : 0.0)) * inv }
        (nl - 1).downto(0) do |l|
          prev = acts[l]
          back = l > 0 ? Array.new(prev.size, 0.0) : nil
          delta.each_with_index do |dv, o|
            gb[l][o] += dv
            row = w[l][o]
            g = gw[l][o]
            prev.each_with_index do |p, i|
              g[i] += dv * p
              back[i] += row[i] * dv if back
            end
          end
          delta = back.each_with_index.map { |v, i| prev[i] > 0 ? v : 0.0 } if back
        end
      end
      t += 1
      c1 = 1 - 0.9**t
      c2 = 1 - 0.999**t
      nl.times do |l|
        w[l].each_with_index do |row, o|
          g_row = gw[l][o]
          m_row = mw[l][o]
          v_row = vw[l][o]
          row.size.times do |i|
            g = g_row[i]
            m = m_row[i] = 0.9 * m_row[i] + 0.1 * g
            v = v_row[i] = 0.999 * v_row[i] + 0.001 * g * g
            row[i] -= lr * (m / c1) / (Math.sqrt(v / c2) + 1e-8)
          end
          g = gb[l][o]
          m = mb[l][o] = 0.9 * mb[l][o] + 0.1 * g
          v = vb[l][o] = 0.999 * vb[l][o] + 0.001 * g * g
          b[l][o] -= lr * (m / c1) / (Math.sqrt(v / c2) + 1e-8)
        end
      end
    end
  end
  [w, b]
end

def mlp_correct(x, y, w, b)
  x.each_with_index.count do |row, i|
    out = mlp_forward(w, b, row)[w.size]
    (out[1] > out[0] ? 1 : 0) == y[i]
  end
end

def logistic_positives(x, w, b)
  x.count do |row|
    z = b
    row.each_with_index { |v, j| z += w[j] * v }
    z >= 0
  end
end

dir, label = ARGV
X = load("#{dir}/cls_X.txt")
LABELS = load("#{dir}/cls_y.txt").map { |r| r[0].to_i }
Q = load("#{dir}/knn_Q.txt")
XL = load("#{dir}/lin_X.txt")
YL = load("#{dir}/lin_y.txt").map { |r| r[0] }
XK = load("#{dir}/km_X.txt")
SEEDS = XK.first(4)
XT = load("#{dir}/tree_X.txt")
YT = load("#{dir}/tree_y.txt").map { |r| r[0].to_i }
QT = load("#{dir}/tree_Q.txt")
W0 = [1, 2, 3].map { |i| load("#{dir}/mlp_W#{i}.txt") }
PERMS = load("#{dir}/mlp_perm.txt").map { |r| r.map(&:to_i) }

bench(label, 'knn', ->(p) { p.sum }) { knn_predict(X, LABELS, Q, 5) }
bench(label, 'logreg', ->(m) { logistic_positives(X, m[0], m[1]) }) { logistic_fit(X, LABELS, 0.5, 200) }
bench(label, 'linreg', ->(w) { format('%.6f', w.sum) }) { linear_fit(XL, YL) }
bench(label, 'kmeans', ->(cs) { format('%.6f', cs.map(&:sum).sum) }) { kmeans(XK, SEEDS, 20) }
bench(label, 'tree', ->(t) { tree_predict(t, QT).sum }) { tree_fit(XT, YT, 10, 3, 1e-7) }
bench(label, 'mlp', ->(m) { mlp_correct(X, LABELS, m[0], m[1]) }) { mlp_fit(X, LABELS, W0, PERMS, 64, 0.005) }
