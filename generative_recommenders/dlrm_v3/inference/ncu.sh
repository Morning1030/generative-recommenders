ncu \
  --target-processes all \
  -f \
  --set full \
  --profile-from-start off \
  --launch-skip 50 \
  --launch-count 50 \
  --kernel-name "_addmm_fwd" \
  -o dlrmv3_addmm \
  python generative_recommenders/dlrm_v3/inference/main.py \
    --dataset sampled-streaming-100b

  # 
  # --nvtx \
  # --nvtx-include "DLRMv3::Official Run" \



  sqlite3 -header -column dlrmv3_nsys.sqlite <<'EOF'
WITH w AS (
  SELECT n.start AS s, n.end AS e
  FROM NVTX_EVENTS n LEFT JOIN StringIds sid ON sid.id = n.textId
  WHERE COALESCE(n.text, sid.value) = 'DLRMv3::Official Run' AND n.end IS NOT NULL
  ORDER BY n.start LIMIT 1),
ev AS (
  SELECT n.globalTid AS tid, n.start AS st, n.end AS en, COALESCE(n.text, sid.value) AS name
  FROM NVTX_EVENTS n LEFT JOIN StringIds sid ON sid.id = n.textId, w
  WHERE n.end IS NOT NULL AND n.start >= w.s AND n.end <= w.e
    AND COALESCE(n.text, sid.value) IN ('after_predict', 'D2H Transfer')),
g AS (
  SELECT a.tid AS tid, a.st AS ge,
         (SELECT MAX(d.en) FROM ev d WHERE d.tid = a.tid AND d.name = 'D2H Transfer' AND d.en <= a.st) AS gs
  FROM ev a WHERE a.name = 'after_predict')
SELECT CASE WHEN ge-gs < 50000 THEN '1: <50us'
            WHEN ge-gs < 1000000 THEN '2: 50us-1ms'
            WHEN ge-gs < 10000000 THEN '3: 1-10ms'
            ELSE '4: >10ms' END AS gap,
       COUNT(*) AS cnt, ROUND(SUM(ge-gs)/1e9, 2) AS total_s,
       ROUND(AVG(ge-gs)/1e6, 2) AS avg_ms, ROUND(MAX(ge-gs)/1e6, 1) AS max_ms
FROM g WHERE gs IS NOT NULL GROUP BY gap ORDER BY gap;
EOF