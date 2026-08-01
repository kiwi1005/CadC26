# 07　GPU 原生析取約束投影：BDP

## 7.1 為何需要析取

對任意矩形 i、j，non-overlap 等價於四者至少一個成立：

\[
R_i\le L_j\ \lor\ R_j\le L_i\ \lor\ T_i\le B_j\ \lor\ T_j\le B_i
\]

這是非凸析取。任何宣稱「單一線性投影就精確解決未選方向的 rectangle non-overlap」都不正確。

BDP 的做法：

1. 只對 overlap/near-conflict pairs 建 active set；
2. direction network 輸出 L/R/T/B logits；
3. 對多組 direction assignments 建 beam；
4. 每個 assignment 形成 convex sparse projection；
5. GPU 平行求解並 exact verify。

## 7.2 Active pair broad phase

N≤120 時可先 dense 計算全部 pair gaps。active 若：

- 正重疊；或
- x/y gap 小於 margin；或
- cluster contact pair；或
- direction network uncertainty 高且可能在 polish 時碰撞。

inactive pair 已有充分 separation，不放入 QP，避免 constraints 過多。

## 7.3 Direction selector

輸入：pair static/dynamic features、scene embedding、candidate residual。輸出：

\[
p_{ij}^{dir}=softmax(z_{ij}^{L,R,T,B})
\]

teacher label 不只用 minimum displacement；對四方向各跑局部快速 projection，評估：

- displacement；
- HPWL delta；
- bbox delta；
- cycle/infeasibility；
- soft constraints delta。

用最小 post-projection cost 方向作 teacher，並保留多個近似最佳方向。

## 7.4 Convex subproblem

固定 shapes 後，variable：

\[
z=[x_1,y_1,\dots,x_N,y_N,L,R,B,T,s_{soft}]
\]

在 sequential convex iteration \(r\) 中：

\[
\min_z \frac12\|z-z_0\|_M^2+g_{wl}^Tz+g_{box}^Tz
+\lambda_s\|s_{soft}\|_1
\]

subject to：

- 已選方向的 pair separation inequalities；
- preplaced center equality；
- bbox containment；
- boundary equality（可帶 soft slack）；
- cluster contact equality + orthogonal interval overlap（soft slack）；
- 其他線性化條件。

HPWL 與 bbox area 可在外層重算 gradient，projection 內保持 convex。hard non-overlap、preplaced 不允許 slack；boundary/grouping 是 soft，可有高代價 slack。

## 7.5 Constraint examples

### i left of j

\[
x_i+\frac{w_i}{2}+\epsilon\le x_j-\frac{w_j}{2}
\]

### bbox containment

\[
L\le x_i-\frac{w_i}{2},\quad x_i+\frac{w_i}{2}\le R
\]

### left boundary

\[
x_i-\frac{w_i}{2}=L
\]

以兩條 inequalities 表示，final snap 再消除浮點殘差。

### cluster abutment（i left j）

\[
x_i+\frac{w_i}{2}=x_j-\frac{w_j}{2}
\]

\[
y_i+\frac{h_i}{2}\ge y_j-\frac{h_j}{2}+\epsilon_c
\]

\[
y_j+\frac{h_j}{2}\ge y_i-\frac{h_i}{2}+\epsilon_c
\]

## 7.6 PDHG solver

將 constraints 寫成 \(Az\le b\)。對 indicator function 的 primal-dual hybrid gradient：

\[
y^{k+1}=\max(0,y^k+\sigma(A\bar z^k-b))
\]

若 primal objective 是 diagonal quadratic + linear term：

\[
z^{k+1}=prox_{\tau f}(z^k-\tau A^Ty^{k+1})
\]

\[
\bar z^{k+1}=z^{k+1}+\theta(z^{k+1}-z^k)
\]

固定 32–64 iterations，適合 CUDA Graph。對 equality 可用兩條 inequalities 或獨立 dual channel。

### 不顯式建 sparse matrix

每個 constraint 只涉及少數 variables，可存：

```text
kind, idx_a, idx_b, coeff_a, coeff_b, rhs, hard_mask
```

`A z` 是 gather/fused expression；`A^T y` 是 scatter-add。這是第一批值得寫 Triton 的 kernel。

## 7.7 Direction beam

- baseline assignment：argmax direction logits；
- minimum-displacement assignment；
- B−2 組用 Sobol/Gumbel 對 top-U uncertain pairs 取樣；
- 先做快速 cycle/infeasibility filter；
- B 組 PDHG 同時求解。

推薦：top candidate 用 B=8；高風險 large case 可 B=16。Population 與 beam 共同形成 batch，不能讓 Python 逐條呼叫 solver。

## 7.8 Cycle 與不可行修復

若 direction inequalities 形成 positive-width cycle，primal residual 不會收斂。處理：

1. 找 residual 最大 constraint/component；
2. 在低 confidence edge 中翻轉方向；
3. 或把部分 constraints 改到另一軸；
4. 重跑有限次；
5. 仍失敗即 discard beam，不修改 hard anchors。

## 7.9 Shape-position alternating projection

外層 2–3 cycles：

1. **Shape step**：更新 soft log-aspect、MIB consensus，精確重建 area。
2. **Position BDP**：固定 shape 解 centers。
3. **Exact residual step**：重建 active pairs、contact/boundary constraints。

這避免在單一非凸 program 同時解 shape product 與 separation。

## 7.10 MIB compatibility

同組要 identical `(w,h)`，等價於 common area 也必須相同。每個 soft block hard area interval：

\[
I_i=[0.99A_i,1.01A_i]
\]

若：

\[
I_g=\bigcap_i I_i\ne\varnothing
\]

選 common area \(A_g\in I_g\) 與 common aspect，MIB 可零 violation且 hard area 合法。若有 fixed/preplaced，required shape 必須落在所有 intervals 才能廣播。若交集為空，hard area 優先，MIB 保留 soft slack；不得為了消除 MIB violation 使 solution infeasible。

## 7.11 Exact finalization

對 BDP candidate：

1. 用 `h=A/w` 重算 soft height；
2. exact overwrite hard targets；
3. 對 selected direction graph 做微小 margin propagation；
4. boundary/contact equality final snap；
5. CPU double 檢查所有 pairs 與 evaluator semantics；
6. 只有通過才進 incumbent。

## 7.12 Deterministic shelf fallback

因 contest 無 fixed outline，可以從零構造 hard-feasible fallback：

1. exact copy preplaced；
2. fixed-shape 使用 target dimensions；
3. free soft blocks 用 square 或相容 MIB common shape；
4. 取所有 preplaced 的最大 right edge，加安全 gap；
5. 其餘 blocks 沿 x 軸依序放成一條 shelf，y 選不與 preplaced 相交的高度；
6. exact verifier；若仍與 preplaced 投影相交，將整條 shelf 再向右移。

此解可能 bbox/HPWL 很差、soft violations 高，但理論上只要官方 instance 的 hard anchors 自身合法，就能避免 cost=10。它完全獨立於任何舊 legalizer。
