# HNSW Sweep Experiment Analysis

## 1. 實驗目的

本實驗使用 SlimPajama 文字資料經由 `sentence-transformers/all-MiniLM-L6-v2` 產生的 embedding，評估 HNSW 在近似最近鄰搜尋上的速度與準確率表現。

主要目標是比較：

- Exact cosine KNN
- HNSW approximate KNN

並觀察不同 HNSW 參數對以下指標的影響：

- `recall@10`
- build time
- query time
- QPS

---

## 2. 實驗設定

本次 sweep 使用以下設定：

```text
N = 20,000
K = 10
embedding model = sentence-transformers/all-MiniLM-L6-v2
similarity metric = cosine
```

HNSW 參數 sweep 範圍：

```text
M = [8, 16, 32]
ef_construction = [100, 200, 400]
ef = [50, 100, 200]
```

其中：

- `M` 控制 HNSW graph 中每個節點的最大連邊數。
- `ef_construction` 控制建圖時的搜尋寬度，通常越大圖品質越好，但建圖越慢。
- `ef` 控制查詢時的搜尋寬度，通常越大 recall 越高，但查詢越慢。

Exact KNN 的時間為：

```text
exact_time_sec = 3.2053 秒
```

---

## 3. 整體結果摘要

HNSW 的 query time 介於：

```text
0.040 ~ 0.420 秒
```

相較於 exact KNN 的：

```text
3.205 秒
```

HNSW 約可達到：

```text
7.6x ~ 80x speedup
```

其中：

- 最準的 HNSW 設定幾乎等同 exact KNN。
- 最快的 HNSW 設定則以較低 recall 換取最高查詢速度。
- 多數設定都能達到相當高的 recall，顯示 HNSW 適合此 embedding search 任務。

---

## 4. 最準設定

最準的設定為：

```text
M = 32
ef_construction = 400
ef = 200
recall@10 = 0.999975
build_time_sec = 0.4108
query_time_sec = 0.4203
qps = 47,585
```

此設定的 `recall@10 = 0.999975`，幾乎等同 exact KNN。

由於本次共有：

```text
20,000 queries × top-10 = 200,000 neighbor results
```

錯誤比例約為：

```text
1 - 0.999975 = 0.000025
```

因此大約只錯：

```text
200,000 × 0.000025 = 5 個 neighbor
```

也就是說，此設定幾乎完全重現 exact KNN 的搜尋結果。

不過這個設定的 query time 也是所有設定中較高的，因此較適合用在需要極高準確率的情境。

---

## 5. 最快設定

最快的設定為：

```text
M = 8
ef_construction = 400
ef = 50
recall@10 = 0.97504
build_time_sec = 0.2038
query_time_sec = 0.0403
qps = 495,810
```

此設定 query time 只有約 0.04 秒，約比 exact KNN 快：

```text
3.205 / 0.0403 ≈ 79.5x
```

但 recall 降到約 97.5%。

以 top-10 來看，平均每筆 query 找回：

```text
10 × 0.97504 = 9.7504
```

也就是每筆 query 平均會漏掉約 0.25 個正確 neighbor。

此設定適合用於追求極高查詢速度、且可以接受少量近鄰誤差的情境。

---

## 6. 推薦的平衡設定

若考慮速度與準確率的平衡，推薦：

```text
M = 16
ef_construction = 200
ef = 100
recall@10 = 0.997965
build_time_sec = 0.1722
query_time_sec = 0.1286
qps = 155,554
```

此設定的優點是：

- `recall@10` 達到約 99.80%
- query time 僅 0.1286 秒
- 約比 exact KNN 快：

```text
3.205 / 0.1286 ≈ 24.9x
```

與更高準確率設定相比，此設定在 recall 只小幅下降的情況下，大幅降低 query time 與建圖成本。

因此，若要在實驗報告中選擇一組代表性的 HNSW 參數，這組設定是一個合理的折衷點。

另一個也可接受的平衡設定為：

```text
M = 16
ef_construction = 100
ef = 100
recall@10 = 0.99599
query_time_sec = 0.1227
qps = 162,964
```

此設定 build time 更低，query time 也略快，但 recall 稍低。

---

## 7. 參數趨勢分析

### 7.1 ef 對 recall 與 query time 影響最明顯

固定：

```text
M = 16
ef_construction = 200
```

不同 `ef` 的結果為：

```text
ef = 50    recall@10 = 0.99403    query_time = 0.0757 sec
ef = 100   recall@10 = 0.997965   query_time = 0.1286 sec
ef = 200   recall@10 = 0.9993     query_time = 0.2239 sec
```

可以觀察到：

- `ef` 越大，recall 越高。
- `ef` 越大，query time 也越長。
- `ef` 是控制查詢階段 accuracy-speed trade-off 的主要參數。

若需要提高 recall，可以優先增加 `ef`。
若需要提高查詢速度，可以降低 `ef`。

---

### 7.2 M 越大，graph 越密，recall 越高

固定：

```text
ef_construction = 100
ef = 50
```

不同 `M` 的結果為：

```text
M = 8    recall@10 = 0.9633    query_time = 0.0404 sec
M = 16   recall@10 = 0.99144   query_time = 0.0705 sec
M = 32   recall@10 = 0.99675   query_time = 0.1075 sec
```

可以觀察到：

- `M` 越大，recall 明顯提升。
- `M` 越大，query time 也會增加。
- 較大的 `M` 會讓圖結構更密，提升搜尋品質，但也增加記憶體與搜尋成本。

在本實驗中，`M=16` 是不錯的折衷點。
`M=8` 查詢最快，但 recall 較低。
`M=32` recall 最高，但速度與建圖成本也較高。

---

### 7.3 ef_construction 主要影響建圖品質，但有邊際效益遞減

固定：

```text
M = 16
ef = 100
```

不同 `ef_construction` 的結果為：

```text
ef_construction = 100   recall@10 = 0.99599    build_time = 0.1060 sec
ef_construction = 200   recall@10 = 0.997965   build_time = 0.1722 sec
ef_construction = 400   recall@10 = 0.998365   build_time = 0.2767 sec
```

可以觀察到：

- 從 100 增加到 200 時，recall 有明顯提升。
- 從 200 增加到 400 時，recall 提升幅度變小。
- build time 隨著 `ef_construction` 增加而明顯上升。

因此 `ef_construction=200` 是較合理的折衷選擇。
若追求最高 recall，可以使用 400；若重視建圖速度，可以使用 100。

---

## 8. 可寫入報告的結論

在 20,000 筆 SlimPajama MiniLM embeddings 上，HNSW 在所有測試設定中皆明顯快於 exact cosine KNN。Exact KNN 需要約 3.21 秒，而 HNSW query time 介於 0.04 至 0.42 秒，約可達到 7.6 至 80 倍的速度提升。

最高準確率設定為 `M=32, ef_construction=400, ef=200`，其 `recall@10` 達到 0.999975，幾乎等同 exact search。然而，此設定的 query time 也較高。

若考慮速度與準確率的平衡，`M=16, ef_construction=200, ef=100` 是較合適的設定。此設定可達到 `recall@10=0.997965`，query time 僅 0.1286 秒，約比 exact KNN 快 25 倍，同時保有非常高的近鄰搜尋品質。

整體而言，HNSW 能在維持高 recall 的同時，大幅降低查詢時間，適合用於大規模 embedding 檢索任務。

---

## 9. 後續實驗建議

下一步可以固定推薦設定：

```text
M = 16
ef_construction = 200
ef = 100
```

然後測試不同資料規模下的 scaling：

```text
N = 10k, 20k, 50k, 100k
```

並比較：

- exact KNN time
- HNSW build time
- HNSW query time
- recall@10
- QPS

這樣可以更清楚觀察 HNSW 相較於 exact KNN 在資料量增加時的擴展性優勢。
