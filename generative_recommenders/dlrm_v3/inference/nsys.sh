WORLD_SIZE=1 \
MALLOC_MMAP_MAX_=0 \
MALLOC_TRIM_THRESHOLD_=4294967296 \
MALLOC_ARENA_MAX=1 \
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
nsys profile \
  -o dlrmv3_nsys_malloc_opt \
  --force-overwrite=true \
  -t cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  --capture-range=nvtx \
  --nvtx-capture="DLRMv3::Official Run" \
  --capture-range-end=stop \
  --stats=true \
  python generative_recommenders/dlrm_v3/inference/main.py \
    --dataset sampled-streaming-100b

WORLD_SIZE=1 \
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
nsys profile \
  -o dlrmv3_nsys \
  --force-overwrite=true \
  -t cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  --capture-range=nvtx \
  --nvtx-capture="DLRMv3::Official Run" \
  --capture-range-end=stop \
  --stats=true \
  python generative_recommenders/dlrm_v3/inference/main.py \
    --dataset sampled-streaming-100b

# Throughput: 87.64
# Collect: 200 us
# 0.0      12,9445,9213       2517         51,4286.5         24,6783.0          3,2201        537,8641      59,8885.7  PushPop   free_cpu_inputs

# 11.7    8765,2454,6505       2517     3,4824,1774.5     3,5440,0303.0       2248,5078     5,3489,4525    6609,9712.5  PushPop   dense predict            
#       7.5    5607,3977,6592       2517     2,2278,0999.8     2,2402,9789.0         27,4194     3,6338,6100    5070,0525.6  PushPop   Wait GPU                 
#       3.2    2369,7299,2895          1  2369,7299,2895.0  2369,7299,2895.0  2369,7299,2895  2369,7299,2895            0.0  PushPop   DLRMv3::Official Run     
#       3.0    2250,9034,2449     2,0000       1125,4517.1            4000.0            2100     5,5342,3388    5093,6370.1  PushPop   issue_queries            
#       3.0    2249,7395,6649       2000     1,1248,6978.3     1,2695,5340.5          3,2880     5,5341,7948  1,2067,6027.3  PushPop   enqueue:dispatch n=10    
#       3.0    2246,9353,3885       1888     1,1901,1405.7     1,2998,4573.5         19,2273     5,5327,9556  1,2092,9260.4  PushPop   producer.put qsize=16    
#       2.2    1631,3746,0140       2517       6481,4247.2       6651,9601.0       1200,2240     1,2615,8604    1387,2149.3  PushPop   dense forward            
#       2.0    1506,1941,3904       2517       5984,0847.8       6160,8482.0        340,0148     1,1721,3738    1409,3158.3  PushPop   dense H2D                
#       0.3     211,0969,5244       2517        838,6847.5        777,1480.0        195,5888       5282,5377     392,6974.4  PushPop   Embedding Lookup         
#       0.1     103,5760,2047       2000        517,8801.0        445,6923.0        280,9350       3650,2406     232,4696.1  PushPop   get_samples              
#       0.0      27,2298,5732     2,0000         13,6149.3          5,9575.5            6990       1681,2748      59,1865.4  StartEnd  query_wait               
#       0.0      22,7777,6908       2517         90,4957.1         53,4348.0          6,4731       1006,8092      93,0222.6  PushPop   postprocess              
#       0.0      11,0073,1615       2000         55,0365.8          1,2931.0            4280     5,6496,7581    1423,4488.5  PushPop   worker.wait_task         
#       0.0       9,2495,2406       2517         36,7482.1         26,1314.0          4,9681        236,4323      26,0114.3  PushPop   D2H Transfer

# nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum dlrmv3_nsys.nsys-rep

WORLD_SIZE=1 \
MALLOC_MMAP_MAX_=0 \
MALLOC_TRIM_THRESHOLD_=4294967296 \
MALLOC_ARENA_MAX=1 \
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
python generative_recommenders/dlrm_v3/inference/main.py \
  --dataset sampled-streaming-100b











# Throughput with setSwitchIntervel=0.0005
# Completed samples per second    : 85.00
#  Time (%)  Total Time (ns)   Instances      Avg (ns)          Med (ns)         Min (ns)        Max (ns)      StdDev (ns)    Style              Range          
#  --------  ----------------  ---------  ----------------  ----------------  --------------  --------------  -------------  --------  -------------------------
#      59.5  3,9769,9527,6666       2000    19,8849,7638.3    19,3038,2763.5         54,8664    29,9926,2946  2,8979,8939.9  StartEnd  task_queue_wait          
#      14.1    9407,5288,4485       2000     4,7037,6442.2     4,0271,3041.0     2,2093,4580    12,1537,0126  1,5916,8632.6  PushPop   worker.batch bs=10       
#       7.3    4849,9095,8305       2517     1,9268,6117.7     1,9828,6868.0        219,7607     3,7137,2824    5737,1383.3  PushPop   Wait GPU                 
#       3.6    2440,0696,6671          1  2440,0696,6671.0  2440,0696,6671.0  2440,0696,6671  2440,0696,6671            0.0  PushPop   DLRMv3::Official Run     
#       3.5    2327,9919,9270     2,0000       1163,9960.0            4030.0            2140     6,6489,3976    5106,4601.8  PushPop   issue_queries            
#       3.5    2326,8391,7390       2000     1,1634,1958.7       9467,8723.5          2,9701     6,6488,4916  1,1789,5215.9  PushPop   enqueue:dispatch n=10    
#       3.5    2323,5641,1033       1949     1,1921,8271.4     1,0361,1890.0         20,1491     6,6457,9213  1,1787,8262.6  PushPop   producer.put qsize=16    
#       2.1    1406,6949,5475       2517       5588,7761.4       5531,5043.0       1132,3611     1,1894,7503    1873,5715.2  PushPop   dense forward            
#       2.0    1325,4579,8229       2517       5266,0229.7       5257,5441.0        452,0706       9845,1589    1367,5386.7  PushPop   dense H2D                
#       0.7     474,4764,3034       2517       1885,0871.3       1613,2829.0        170,6084     1,0431,3365    1086,3097.9  PushPop   Embedding Lookup         
#       0.2     122,4916,1715       2000        612,4580.9        390,0516.5        234,1929       6645,7422     541,5125.9  PushPop   get_samples              
#       0.1      63,7615,1031       2517        253,3234.4         36,7963.0          6,5550       4617,2900     553,9659.0  PushPop   postprocess              
#       0.0      22,0868,4401     2,0000         11,0434.2          5,0545.5            6990       1725,9839      60,6452.5  StartEnd  query_wait               
#       0.0      10,6961,2591       2000         53,4806.3          1,3210.5            6260     4,5613,7514    1274,8756.8  PushPop   worker.wait_task         
#       0.0       8,8998,1273       2517         35,3588.1         28,3482.0          6,2560        318,2665      19,6760.1  PushPop   D2H Transfer      


# Throughput with setSwitchIntervel=0.0005
# Completed samples per second    : 82.31
# Time (%)  Total Time (ns)   Instances      Avg (ns)          Med (ns)         Min (ns)        Max (ns)      StdDev (ns)    Style              Range          
#  --------  ----------------  ---------  ----------------  ----------------  --------------  --------------  -------------  --------  -------------------------
#      59.5  4,0975,4818,7251       2000    20,4877,4093.6    19,1951,3355.5         43,1423    31,7796,4904  3,8584,2026.9  StartEnd  task_queue_wait          
#      14.1    9712,8033,5455       2000     4,8564,0167.7     4,2248,3730.5     2,1878,3523    11,9065,4866  1,6676,8125.7  PushPop   worker.batch bs=10       
#       7.3    5050,2104,7739       2517     2,0064,4039.6     1,9799,2278.0          5,4980     3,7180,0662    7033,5348.2  PushPop   Wait GPU                 
#       3.7    2517,3327,5482          1  2517,3327,5482.0  2517,3327,5482.0  2517,3327,5482  2517,3327,5482            0.0  PushPop   DLRMv3::Official Run     
#       3.5    2393,2993,1966     2,0000       1196,6496.6            3810.0            2080     6,6094,8017    5601,1644.6  PushPop   issue_queries            
#       3.5    2392,2077,5602       2000     1,1961,0387.8       8055,9373.5          3,4451     6,6094,2737  1,3603,0732.0  PushPop   enqueue:dispatch n=10    
#       3.5    2389,1536,3353       1960     1,2189,5593.5       8415,5620.0         18,4681     6,6082,2696  1,3629,7375.2  PushPop   producer.put qsize=16    
#       1.9    1342,0886,8825       2517       5332,0965.0       5314,3069.0        924,2996     1,2818,7979    1827,9910.3  PushPop   dense forward            
#       1.9    1294,5256,5578       2517       5143,1293.4       5220,6033.0        293,0588     1,0773,0296    1432,4505.4  PushPop   dense H2D                
#       0.8     538,7216,2457       2517       2140,3322.4       1692,1251.0        186,9501     1,5607,5277    1540,5131.2  PushPop   Embedding Lookup         
#       0.2     143,7599,3556       2000        718,7996.8        419,1870.0        241,7484       9257,7336     757,3269.3  PushPop   get_samples              
#       0.1      68,6386,4817       2517        272,7002.3         36,9872.0          6,5150       4945,4176     676,0580.9  PushPop   postprocess              
#       0.0      21,4305,5140     2,0000         10,7152.8          4,8355.0            7490       1726,5924      60,5689.5  StartEnd  query_wait               
#       0.0      11,7351,8386       2000         58,6759.2          1,3185.0            4029     6,3787,9309    1569,5747.3  PushPop   worker.wait_task         
#       0.0       8,3533,6318       2517         33,1877.8         26,9042.0          5,7250        558,0073      20,8877.9  PushPop   D2H Transfer             
#       0.0       4,9504,3682       5034          9,8340.0          9,3730.0          4,8260        311,2628       5,4249.1  PushPop   CCCL:cub::DeviceRadixSort
#       0.0       3,2195,1738     2,0000          1,6097.6            5040.0            2380        501,1440       7,3373.7  PushPop   QuerySamplesComplete     

# Throughput with setSwitchIntervel=0.005
# Completed samples per second    : 70.55
#  59.7  4,7912,3770,6636       2000    23,9561,8853.3    24,4645,5033.0         48,5916    33,4574,6398  4,6262,0847.3  StartEnd  task_queue_wait          
#      14.1  1,1332,1017,0662       2000     5,6660,5085.3     5,6959,0939.5     2,1259,3336    12,3868,5943  1,9269,7361.7  PushPop   worker.batch bs=10       
#       7.1    5690,4594,1927       2517     2,2608,1025.8     2,5375,3758.0          3,6291     3,7739,9077    8965,3627.9  PushPop   Wait GPU                 
#       3.6    2922,3148,3925          1  2922,3148,3925.0  2922,3148,3925.0  2922,3148,3925  2922,3148,3925            0.0  PushPop   DLRMv3::Official Run     
#       3.5    2803,7577,9883     2,0000       1401,8789.0            4030.0            2090     6,1749,6689    7119,0013.5  PushPop   issue_queries            
#       3.5    2802,6199,0005       2000     1,4013,0995.0       4575,5448.5          2,9210     6,1748,9769  1,8171,6037.4  PushPop   enqueue:dispatch n=10    
#       3.5    2797,9886,2330       1943     1,4400,3531.8       4826,1269.0         18,9302     6,1728,7897  1,8270,9718.1  PushPop   producer.put qsize=16    
#       1.8    1430,4466,5456       2517       5683,1412.6       5746,6242.0        298,5547     1,5347,9594    2000,5639.5  PushPop   dense H2D                
#       1.7    1373,8962,4520       2517       5458,4674.0       5826,1301.0        909,2371     1,2816,0155    1781,6758.6  PushPop   dense forward            
#       1.0     796,9356,4223       2517       3166,2123.3       2293,0350.0        162,8120     3,0719,2971    2990,4432.1  PushPop   Embedding Lookup         
#       0.3     200,9589,3215       2000       1004,7946.6        519,7468.0        235,8349       9614,1544    1150,9458.0  PushPop   get_samples              
#       0.1     101,9358,7124       2517        404,9895.6         39,2914.0          4,9020       6435,2405     956,9786.3  PushPop   postprocess              
#       0.0      23,4343,3237     2,0000         11,7171.7          4,9565.5            6920       1729,0561      62,2308.0  StartEnd  query_wait               
#       0.0      12,3704,3378       2000         61,8521.7          1,3860.0            4761     4,7873,1355    1475,6613.0  PushPop   worker.wait_task         
#       0.0       7,8679,0014       2517         31,2590.4         24,9823.0          4,8590        802,7898      31,4128.1  PushPop   D2H Transfer             
#       0.0       5,3587,3045       5034         10,6450.7          9,2331.5          4,4991        984,2630      19,6580.1  PushPop   CCCL:cub::DeviceRadixSort
#       0.0       4,2595,7477     2,0000          2,1297.9            5630.0            2560        767,0414      13,8162.5  PushPop   QuerySamplesComplete  