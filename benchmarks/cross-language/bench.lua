-- Usage: lua bench.lua <data dir> <label>
-- Times with os.clock() (CPU time of this process; single-threaded, so ~= wall time).

local function load(path)
  local rows = {}
  local first = true
  for line in io.lines(path) do
    if first then
      first = false
    elseif line:match('%S') then
      local row = {}
      for v in line:gmatch('%S+') do row[#row + 1] = tonumber(v) end
      rows[#rows + 1] = row
    end
  end
  return rows
end

local function bench(label, name, fn, check)
  local repeats = tonumber(os.getenv('REPEATS') or '5')
  local result = fn() -- warm-up
  local times = {}
  for r = 1, repeats do
    local t = os.clock()
    result = fn()
    times[r] = (os.clock() - t) * 1000
  end
  table.sort(times)
  print(string.format('%s|%s|%.3f|%.3f|%s', label, name, times[repeats // 2 + 1], times[1], check(result)))
  io.stdout:flush()
end

local function knn_predict(X, labels, queries, k)
  local out = {}
  for qi = 1, #queries do
    local q = queries[qi]
    local dist = {}
    for i = 1, #X do
      local row = X[i]
      local s = 0.0
      for j = 1, #row do
        local diff = row[j] - q[j]
        s = s + diff * diff
      end
      dist[i] = { s, i }
    end
    table.sort(dist, function(a, b) return a[1] < b[1] or (a[1] == b[1] and a[2] < b[2]) end)
    local votes = {}
    for n = 1, k do
      local l = labels[dist[n][2]]
      votes[l] = (votes[l] or 0) + 1
    end
    local best, bestVotes = nil, -1
    for l, v in pairs(votes) do
      if v > bestVotes then best, bestVotes = l, v end
    end
    out[qi] = best
  end
  return out
end

local function logistic_fit(X, y, lr, iterations)
  local d, n = #X[1], #X
  local w = {}
  for j = 1, d do w[j] = 0.0 end
  local b = 0.0
  for _ = 1, iterations do
    local gw = {}
    for j = 1, d do gw[j] = 0.0 end
    local gb = 0.0
    for i = 1, n do
      local row = X[i]
      local z = b
      for j = 1, d do z = z + w[j] * row[j] end
      local err = 1.0 / (1.0 + math.exp(-z)) - y[i]
      gb = gb + err
      for j = 1, d do gw[j] = gw[j] + err * row[j] end
    end
    for j = 1, d do w[j] = w[j] - lr * gw[j] / n end
    b = b - lr * gb / n
  end
  return { w, b }
end

local function linear_fit(X, y)
  local d = #X[1] + 1
  local A, rhs = {}, {}
  for a = 1, d do
    A[a] = {}
    for c = 1, d do A[a][c] = 0.0 end
    rhs[a] = 0.0
  end
  for i = 1, #X do
    local row = {}
    for j = 1, d - 1 do row[j] = X[i][j] end
    row[d] = 1.0
    for a = 1, d do
      rhs[a] = rhs[a] + row[a] * y[i]
      local Aa, ra = A[a], row[a]
      for c = 1, d do Aa[c] = Aa[c] + ra * row[c] end
    end
  end
  for p = 1, d do
    for r = p + 1, d do
      local f = A[r][p] / A[p][p]
      for c = p, d do A[r][c] = A[r][c] - f * A[p][c] end
      rhs[r] = rhs[r] - f * rhs[p]
    end
  end
  local w = {}
  for r = d, 1, -1 do
    local s = rhs[r]
    for c = r + 1, d do s = s - A[r][c] * w[c] end
    w[r] = s / A[r][r]
  end
  return w
end

local function kmeans(X, seeds, iterations)
  local k, d = #seeds, #X[1]
  local centroids = {}
  for c = 1, k do
    centroids[c] = {}
    for j = 1, d do centroids[c][j] = seeds[c][j] end
  end
  for _ = 1, iterations do
    local sums, counts = {}, {}
    for c = 1, k do
      sums[c] = {}
      for j = 1, d do sums[c][j] = 0.0 end
      counts[c] = 0
    end
    for i = 1, #X do
      local x = X[i]
      local best, bestDist = 1, math.huge
      for c = 1, k do
        local center = centroids[c]
        local dist = 0.0
        for j = 1, d do
          local diff = x[j] - center[j]
          dist = dist + diff * diff
        end
        if dist < bestDist then bestDist, best = dist, c end
      end
      counts[best] = counts[best] + 1
      local s = sums[best]
      for j = 1, d do s[j] = s[j] + x[j] end
    end
    for c = 1, k do
      if counts[c] > 0 then
        for j = 1, d do centroids[c][j] = sums[c][j] / counts[c] end
      end
    end
  end
  return centroids
end

local function sum(t)
  local s = 0
  for i = 1, #t do s = s + t[i] end
  return s
end

-- CART with Gini impurity on two classes; nodes are {feature, threshold, left, right, class}, 1-based.
local function tree_fit(X, y, max_height, max_leaf, min_purity)
  local d = #X[1]
  local all = {}
  for i = 1, #X do all[i] = i end
  local nodes = { false }
  local stack = { { 1, all, 0 } }
  while #stack > 0 do
    local task = table.remove(stack)
    local id, rows, depth = task[1], task[2], task[3]
    local n = #rows
    local c1 = 0
    for i = 1, n do c1 = c1 + y[rows[i]] end
    local c0 = n - c1
    local impurity = n - (c0 * c0 + c1 * c1) / n
    local leaf = { 0, 0.0, 0, 0, c1 > c0 and 1 or 0 }
    local split = false
    if not (depth >= max_height or n <= max_leaf or impurity <= 1e-12) then
      local best_gain, best_f, best_t = 0.0, 0, 0.0
      for f = 1, d do
        local order = {}
        for i = 1, n do order[i] = rows[i] end
        table.sort(order, function(a, b) return X[a][f] < X[b][f] end)
        local v = {}
        for i = 1, n do v[i] = X[order[i]][f] end
        if v[n] > v[1] then
          local l0, l1 = 0, 0
          for i = 1, n - 1 do
            if y[order[i]] == 1 then l1 = l1 + 1 else l0 = l0 + 1 end
            if v[i] ~= v[i + 1] then
              local ln = i
              local rn = n - ln
              local r0, r1 = c0 - l0, c1 - l1
              local gain = impurity - (ln - (l0 * l0 + l1 * l1) / ln) - (rn - (r0 * r0 + r1 * r1) / rn)
              if gain > best_gain then
                best_gain, best_f, best_t = gain, f, v[i] + (v[i + 1] - v[i]) / 2
              end
            end
          end
        end
      end
      if best_f > 0 and best_gain / n >= min_purity then
        local left, right = {}, {}
        for i = 1, n do
          local r = rows[i]
          if X[r][best_f] < best_t then left[#left + 1] = r else right[#right + 1] = r end
        end
        if #left > 0 and #right > 0 then
          local l = #nodes + 1
          nodes[l], nodes[l + 1] = false, false
          nodes[id] = { best_f, best_t, l, l + 1, 0 }
          stack[#stack + 1] = { l + 1, right, depth + 1 }
          stack[#stack + 1] = { l, left, depth + 1 }
          split = true
        end
      end
    end
    if not split then nodes[id] = leaf end
  end
  return nodes
end

local function tree_predict(nodes, queries)
  local out = {}
  for qi = 1, #queries do
    local q, i = queries[qi], 1
    while nodes[i][1] > 0 do
      if q[nodes[i][1]] < nodes[i][2] then i = nodes[i][3] else i = nodes[i][4] end
    end
    out[qi] = nodes[i][5]
  end
  return out
end

-- Dense ReLU layers, linear output; W[l] is out x in. Returns every layer's activations.
local function mlp_forward(W, b, x)
  local acts = { x }
  local L = #W
  for l = 1, L do
    local a = {}
    local Wl, bl = W[l], b[l]
    for o = 1, #Wl do
      local row = Wl[o]
      local z = bl[o]
      for i = 1, #row do z = z + row[i] * x[i] end
      if l < L and z < 0 then z = 0.0 end
      a[o] = z
    end
    acts[l + 1] = a
    x = a
  end
  return acts
end

local function zeros2(W)
  local out = {}
  for l = 1, #W do
    out[l] = {}
    for o = 1, #W[l] do
      local r = {}
      for i = 1, #W[l][o] do r[i] = 0.0 end
      out[l][o] = r
    end
  end
  return out
end

local function zeros1(W)
  local out = {}
  for l = 1, #W do
    out[l] = {}
    for o = 1, #W[l] do out[l][o] = 0.0 end
  end
  return out
end

local function mlp_fit(X, y, W0, perms, batch, lr)
  local W = zeros2(W0)
  for l = 1, #W0 do
    for o = 1, #W0[l] do
      for i = 1, #W0[l][o] do W[l][o][i] = W0[l][o][i] end
    end
  end
  local L = #W
  local b, mW, vW, mb, vb = zeros1(W), zeros2(W), zeros2(W), zeros1(W), zeros1(W)
  local t = 0
  for _, perm in ipairs(perms) do
    for s = 1, #perm, batch do
      local last = math.min(#perm, s + batch - 1)
      local inv = 1.0 / (last - s + 1)
      local gW, gb = zeros2(W), zeros1(W)
      for p = s, last do
        local r = perm[p] + 1
        local acts = mlp_forward(W, b, X[r])
        local out = acts[L + 1]
        local mx = math.max(out[1], out[2])
        local e1, e2 = math.exp(out[1] - mx), math.exp(out[2] - mx)
        local tot = e1 + e2
        local delta = { (e1 / tot - (y[r] == 0 and 1.0 or 0.0)) * inv, (e2 / tot - (y[r] == 1 and 1.0 or 0.0)) * inv }
        for l = L, 1, -1 do
          local prev = acts[l]
          local back = nil
          if l > 1 then
            back = {}
            for i = 1, #prev do back[i] = 0.0 end
          end
          for o = 1, #delta do
            local dv = delta[o]
            gb[l][o] = gb[l][o] + dv
            local row, g = W[l][o], gW[l][o]
            for i = 1, #prev do
              g[i] = g[i] + dv * prev[i]
              if back then back[i] = back[i] + row[i] * dv end
            end
          end
          if back then
            for i = 1, #back do
              if not (prev[i] > 0) then back[i] = 0.0 end
            end
            delta = back
          end
        end
      end
      t = t + 1
      local c1, c2 = 1 - 0.9 ^ t, 1 - 0.999 ^ t
      for l = 1, L do
        for o = 1, #W[l] do
          local row, gRow, mRow, vRow = W[l][o], gW[l][o], mW[l][o], vW[l][o]
          for i = 1, #row do
            local g = gRow[i]
            local m = 0.9 * mRow[i] + 0.1 * g
            local v = 0.999 * vRow[i] + 0.001 * g * g
            mRow[i], vRow[i] = m, v
            row[i] = row[i] - lr * (m / c1) / (math.sqrt(v / c2) + 1e-8)
          end
          local g = gb[l][o]
          local m = 0.9 * mb[l][o] + 0.1 * g
          local v = 0.999 * vb[l][o] + 0.001 * g * g
          mb[l][o], vb[l][o] = m, v
          b[l][o] = b[l][o] - lr * (m / c1) / (math.sqrt(v / c2) + 1e-8)
        end
      end
    end
  end
  return { W, b }
end

local function mlp_correct(X, y, W, b)
  local n = 0
  for i = 1, #X do
    local out = mlp_forward(W, b, X[i])[#W + 1]
    if (out[2] > out[1] and 1 or 0) == y[i] then n = n + 1 end
  end
  return n
end

local function logistic_positives(X, w, b)
  local n = 0
  for i = 1, #X do
    local z = b
    for j = 1, #X[i] do z = z + w[j] * X[i][j] end
    if z >= 0 then n = n + 1 end
  end
  return n
end

local dir, label = arg[1], arg[2]
local X = load(dir .. '/cls_X.txt')
local labels = {}
for i, r in ipairs(load(dir .. '/cls_y.txt')) do labels[i] = math.tointeger(r[1]) or r[1] end
local Q = load(dir .. '/knn_Q.txt')
local Xl = load(dir .. '/lin_X.txt')
local yl = {}
for i, r in ipairs(load(dir .. '/lin_y.txt')) do yl[i] = r[1] end
local Xk = load(dir .. '/km_X.txt')
local seeds = { Xk[1], Xk[2], Xk[3], Xk[4] }
local Xt = load(dir .. '/tree_X.txt')
local yt = {}
for i, r in ipairs(load(dir .. '/tree_y.txt')) do yt[i] = math.tointeger(r[1]) end
local Qt = load(dir .. '/tree_Q.txt')
local W0 = { load(dir .. '/mlp_W1.txt'), load(dir .. '/mlp_W2.txt'), load(dir .. '/mlp_W3.txt') }
local perms = {}
for i, r in ipairs(load(dir .. '/mlp_perm.txt')) do
  perms[i] = {}
  for j, v in ipairs(r) do perms[i][j] = math.tointeger(v) end
end

bench(label, 'knn', function() return knn_predict(X, labels, Q, 5) end, function(p) return string.format('%d', sum(p)) end)
bench(label, 'logreg', function() return logistic_fit(X, labels, 0.5, 200) end,
  function(m) return logistic_positives(X, m[1], m[2]) end)
bench(label, 'linreg', function() return linear_fit(Xl, yl) end, function(w) return string.format('%.6f', sum(w)) end)
bench(label, 'kmeans', function() return kmeans(Xk, seeds, 20) end, function(cs)
  local s = 0
  for c = 1, #cs do s = s + sum(cs[c]) end
  return string.format('%.6f', s)
end)
bench(label, 'tree', function() return tree_fit(Xt, yt, 10, 3, 1e-7) end,
  function(t) return string.format('%d', sum(tree_predict(t, Qt))) end)
bench(label, 'mlp', function() return mlp_fit(X, labels, W0, perms, 64, 0.005) end,
  function(m) return mlp_correct(X, labels, m[1], m[2]) end)
