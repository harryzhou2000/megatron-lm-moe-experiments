# Optimization Progress

## First optimizations

### topk: radix selection instead

#### Results

```log
Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128     128    4  softmax  False      0  float32 forward    0.0133    0.0325    2.45x      9641342
aux_loss      512     128    4  softmax  False      0  float32 forward    0.0144    0.0379    2.64x     35673674
aux_loss     2048     128    4  softmax  False      0  float32 forward    0.0288    0.0561    1.95x     71077939
aux_loss     8192     128    4  softmax  False      0  float32 forward    0.0942    0.1389    1.47x     86963906
aux_loss    32768     128    4  softmax  False      0  float32 forward    0.3440    0.4510    1.31x     95259887
aux_loss   131072     128    4  softmax  False      0  float32 forward    1.2057    1.4593    1.21x    108710135
```

```log
Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128    2304   36  softmax  False      0  float32 forward    0.0592    0.0716    1.21x      2161718
aux_loss      512    2304   36  softmax  False      0  float32 forward    0.0664    0.1742    2.62x      7715603
aux_loss     2048    2304   36  softmax  False      0  float32 forward    0.1475    0.3208    2.17x     13880334
aux_loss     8192    2304   36  softmax  False      0  float32 forward    0.4416    0.9869    2.23x     18551571
aux_loss    32768    2304   36  softmax  False      0  float32 forward    1.3322    3.4979    2.63x     24597355
aux_loss   131072    2304   36  softmax  False      0  float32 forward    5.3452   14.1751    2.65x     24521341

```

```log
Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128     128    4  softmax   True      0  float32 forward    0.0172    0.0524    3.05x      7461990
    topk      512     128    4  softmax   True      0  float32 forward    0.0169    0.0532    3.16x     30372635
    topk     2048     128    4  softmax   True      0  float32 forward    0.0288    0.0723    2.51x     71003031
    topk     8192     128    4  softmax   True      0  float32 forward    0.0942    0.1576    1.67x     86986362
    topk    32768     128    4  softmax   True      0  float32 forward    0.3509    0.4286    1.22x     93383796
    topk   131072     128    4  softmax   True      0  float32 forward    1.1089    1.5368    1.39x    118201840
```

```log
Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128    2304   36  softmax   True      0  float32 forward    0.0477    0.0711    1.49x      2682529
    topk      512    2304   36  softmax   True      0  float32 forward    0.0536    0.1599    2.98x      9550528
    topk     2048    2304   36  softmax   True      0  float32 forward    0.1214    0.2969    2.44x     16864784
    topk     8192    2304   36  softmax   True      0  float32 forward    0.3648    1.0261    2.81x     22453678
    topk    32768    2304   36  softmax   True      0  float32 forward    1.3793    3.8182    2.77x     23757014
    topk   131072    2304   36  softmax   True      0  float32 forward    5.5381   15.4401    2.79x     23667278
```

```log
Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=backward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128    2304   36  softmax  False      0  float32 backward    0.1025    0.0710    0.69x      1249301
aux_loss      512    2304   36  softmax  False      0  float32 backward    0.0970    0.0715    0.74x      5279918
aux_loss     2048    2304   36  softmax  False      0  float32 backward    0.1449    0.0890    0.61x     14136992
aux_loss     8192    2304   36  softmax  False      0  float32 backward    0.3625    0.2541    0.70x     22596058
aux_loss    32768    2304   36  softmax  False      0  float32 backward    1.2745    0.9312    0.73x     25710207
aux_loss   131072    2304   36  softmax  False      0  float32 backward    4.9311    3.6631    0.74x     26580928
```

```log
Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=backward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128    2304   36  softmax   True      0  float32 backward    0.1048    0.1118    1.07x      1221769
    topk      512    2304   36  softmax   True      0  float32 backward    0.1067    0.1053    0.99x      4800250
    topk     2048    2304   36  softmax   True      0  float32 backward    0.2382    0.1162    0.49x      8599296
    topk     8192    2304   36  softmax   True      0  float32 backward    0.6750    0.2963    0.44x     12137154
    topk    32768    2304   36  softmax   True      0  float32 backward    2.5405    1.0521    0.41x     12898395
    topk   131072    2304   36  softmax   True      0  float32 backward    9.9940    4.1019    0.41x     13115071
```

#### main results

```log
Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128     128    4  softmax  False      0  float32 forward    0.0132    0.0335    2.55x      9731177
aux_loss      512     128    4  softmax  False      0  float32 forward    0.0127    0.0390    3.07x     40301252
aux_loss     2048     128    4  softmax  False      0  float32 forward    0.0124    0.0558    4.49x    164880456
aux_loss     8192     128    4  softmax  False      0  float32 forward    0.0247    0.1387    5.61x    331417333
aux_loss    32768     128    4  softmax  False      0  float32 forward    0.0805    0.4533    5.63x    406982267
aux_loss   131072     128    4  softmax  False      0  float32 forward    0.3085    1.5472    5.01x    424826637

```

```log
Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128    2304   36  softmax  False      0  float32 forward    2.0998    0.0575    0.03x        60959
aux_loss      512    2304   36  softmax  False      0  float32 forward    1.9347    0.1416    0.07x       264645
aux_loss     2048    2304   36  softmax  False      0  float32 forward    3.9439    0.2744    0.07x       519285
aux_loss     8192    2304   36  softmax  False      0  float32 forward   11.9394    0.9339    0.08x       686130
aux_loss    32768    2304   36  softmax  False      0  float32 forward   42.5505    3.5050    0.08x       770097
aux_loss   131072    2304   36  softmax  False      0  float32 forward  167.8365   14.1997    0.08x       780950
```

```log
Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128     128    4  softmax   True      0  float32 forward    0.0172    0.0519    3.02x      7440753
    topk      512     128    4  softmax   True      0  float32 forward    0.0165    0.0530    3.22x     31080634
    topk     2048     128    4  softmax   True      0  float32 forward    0.0165    0.0719    4.35x    123774345
    topk     8192     128    4  softmax   True      0  float32 forward    0.0251    0.1575    6.27x    326368272
    topk    32768     128    4  softmax   True      0  float32 forward    0.0880    0.4832    5.49x    372218814
    topk   131072     128    4  softmax   True      0  float32 forward    0.2874    1.5447    5.37x    456039905
```

```log
Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128    2304   36  softmax   True      0  float32 forward    1.8890    0.0696    0.04x        67762
    topk      512    2304   36  softmax   True      0  float32 forward    1.9133    0.1598    0.08x       267602
    topk     2048    2304   36  softmax   True      0  float32 forward    3.8982    0.2970    0.08x       525370
    topk     8192    2304   36  softmax   True      0  float32 forward   11.8386    1.0299    0.09x       691975
    topk    32768    2304   36  softmax   True      0  float32 forward   42.1016    3.8291    0.09x       778308
    topk   131072    2304   36  softmax   True      0  float32 forward  165.9964   15.4713    0.09x       789608
```

#### Machine #2, after cherry-pick

```log
========================================================================
Device 0: NVIDIA B300 SXM6 AC
  Compute capability : 10.3
  SMs                : 148
  Global memory      : 267.7 GiB
  (shared mem info unavailable)
  CUDA version       : 13.1
  PyTorch version    : 2.11.0a0+eb65b36914.nv26.02
========================================================================

Running 6 aux_loss correctness test(s) with user config (dtype=torch.float32, pass=forward)...

  PASS [aux_loss forward | softmax | tokens=     1 | experts= 128 | topk=4 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=    64 | experts= 128 | topk=4 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=   512 | experts= 128 | topk=4 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=  2048 | experts= 128 | topk=4 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=  8192 | experts= 128 | topk=4 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens= 32768 | experts= 128 | topk=4 | dtype=torch.float32 | input=arange]

Aux loss correctness: 6/6 passed


Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128     128    4  softmax  False      0  float32 forward    0.0167    0.0391    2.35x      7682557
aux_loss      512     128    4  softmax  False      0  float32 forward    0.0165    0.0384    2.32x     30944184
aux_loss     2048     128    4  softmax  False      0  float32 forward    0.0183    0.0439    2.40x    112062477
aux_loss     8192     128    4  softmax  False      0  float32 forward    0.0476    0.0779    1.64x    171977135
aux_loss    32768     128    4  softmax  False      0  float32 forward    0.1619    0.2154    1.33x    202441563
aux_loss   131072     128    4  softmax  False      0  float32 forward    0.6134    0.8174    1.33x    213680789

========================================================================
Device 0: NVIDIA B300 SXM6 AC
  Compute capability : 10.3
  SMs                : 148
  Global memory      : 267.7 GiB
  (shared mem info unavailable)
  CUDA version       : 13.1
  PyTorch version    : 2.11.0a0+eb65b36914.nv26.02
========================================================================

Running 6 aux_loss correctness test(s) with user config (dtype=torch.float32, pass=forward)...

  PASS [aux_loss forward | softmax | tokens=     1 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=    64 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=   512 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=  2048 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens=  8192 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss forward | softmax | tokens= 32768 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]

Aux loss correctness: 6/6 passed


Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128    2304   36  softmax  False      0  float32 forward    0.0627    0.0503    0.80x      2042254
aux_loss      512    2304   36  softmax  False      0  float32 forward    0.0616    0.0978    1.59x      8311472
aux_loss     2048    2304   36  softmax  False      0  float32 forward    0.1021    0.2086    2.04x     20055027
aux_loss     8192    2304   36  softmax  False      0  float32 forward    0.3861    0.5716    1.48x     21215727
aux_loss    32768    2304   36  softmax  False      0  float32 forward    1.5466    2.1288    1.38x     21187678
aux_loss   131072    2304   36  softmax  False      0  float32 forward    6.1951    8.5068    1.37x     21157205

========================================================================
Device 0: NVIDIA B300 SXM6 AC
  Compute capability : 10.3
  SMs                : 148
  Global memory      : 267.7 GiB
  (shared mem info unavailable)
  CUDA version       : 13.1
  PyTorch version    : 2.11.0a0+eb65b36914.nv26.02
========================================================================

Running 6 topk correctness test(s) with user config (dtype=torch.float32, pass=forward)...

  PASS [topk forward | softmax | tokens=     1 | experts= 128 | topk=4 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=    64 | experts= 128 | topk=4 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=   512 | experts= 128 | topk=4 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=  2048 | experts= 128 | topk=4 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=  8192 | experts= 128 | topk=4 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens= 32768 | experts= 128 | topk=4 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]

Topk correctness: 6/6 passed


Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128     128    4  softmax   True      0  float32 forward    0.0175    0.0603    3.43x      7294212
    topk      512     128    4  softmax   True      0  float32 forward    0.0185    0.0575    3.11x     27665387
    topk     2048     128    4  softmax   True      0  float32 forward    0.0186    0.0620    3.33x    110116997
    topk     8192     128    4  softmax   True      0  float32 forward    0.0501    0.0983    1.96x    163578278
    topk    32768     128    4  softmax   True      0  float32 forward    0.1673    0.2424    1.45x    195864661
    topk   131072     128    4  softmax   True      0  float32 forward    0.6322    0.8719    1.38x    207342430

========================================================================
Device 0: NVIDIA B300 SXM6 AC
  Compute capability : 10.3
  SMs                : 148
  Global memory      : 267.7 GiB
  (shared mem info unavailable)
  CUDA version       : 13.1
  PyTorch version    : 2.11.0a0+eb65b36914.nv26.02
========================================================================

Running 6 topk correctness test(s) with user config (dtype=torch.float32, pass=forward)...

  PASS [topk forward | softmax | tokens=     1 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=    64 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=   512 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=  2048 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens=  8192 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk forward | softmax | tokens= 32768 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]

Topk correctness: 6/6 passed


Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=forward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128    2304   36  softmax   True      0  float32 forward    0.0699    0.0687    0.98x      1830463
    topk      512    2304   36  softmax   True      0  float32 forward    0.0698    0.1185    1.70x      7335277
    topk     2048    2304   36  softmax   True      0  float32 forward    0.1069    0.2360    2.21x     19166440
    topk     8192    2304   36  softmax   True      0  float32 forward    0.3580    0.6370    1.78x     22883643
    topk    32768    2304   36  softmax   True      0  float32 forward    1.5512    2.3171    1.49x     21124774
    topk   131072    2304   36  softmax   True      0  float32 forward    6.3070    9.2130    1.46x     20781839

========================================================================
Device 0: NVIDIA B300 SXM6 AC
  Compute capability : 10.3
  SMs                : 148
  Global memory      : 267.7 GiB
  (shared mem info unavailable)
  CUDA version       : 13.1
  PyTorch version    : 2.11.0a0+eb65b36914.nv26.02
========================================================================

Running 6 aux_loss correctness test(s) with user config (dtype=torch.float32, pass=backward)...

  PASS [aux_loss backward | softmax | tokens=     1 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss backward | softmax | tokens=    64 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss backward | softmax | tokens=   512 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss backward | softmax | tokens=  2048 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss backward | softmax | tokens=  8192 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]
  PASS [aux_loss backward | softmax | tokens= 32768 | experts=2304 | topk=36 | dtype=torch.float32 | input=arange]

Aux loss correctness: 6/6 passed


Benchmarking 6 aux_loss config(s) (warmup=20, iters=100, dtype=torch.float32, pass=backward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
aux_loss      128    2304   36  softmax  False      0  float32 backward    0.4073    0.3906    0.96x       314278
aux_loss      512    2304   36  softmax  False      0  float32 backward    0.4299    0.3109    0.72x      1191057
aux_loss     2048    2304   36  softmax  False      0  float32 backward    0.4464    0.3631    0.81x      4587380
aux_loss     8192    2304   36  softmax  False      0  float32 backward    0.4800    0.3966    0.83x     17066212
aux_loss    32768    2304   36  softmax  False      0  float32 backward    1.0340    0.7279    0.70x     31690777
aux_loss   131072    2304   36  softmax  False      0  float32 backward    3.2573    2.2048    0.68x     40240069

========================================================================
Device 0: NVIDIA B300 SXM6 AC
  Compute capability : 10.3
  SMs                : 148
  Global memory      : 267.7 GiB
  (shared mem info unavailable)
  CUDA version       : 13.1
  PyTorch version    : 2.11.0a0+eb65b36914.nv26.02
========================================================================

Running 6 topk correctness test(s) with user config (dtype=torch.float32, pass=backward)...

  PASS [topk backward | softmax | tokens=     1 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk backward | softmax | tokens=    64 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk backward | softmax | tokens=   512 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  PASS [topk backward | softmax | tokens=  2048 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
  FAIL [topk backward | softmax | tokens=  8192 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
       grad max abs diff  : 4.842388e-04
       grad mean abs diff : 1.421888e-08
       Tensor-likes are not close!

Mismatched elements: 554 / 18874368 (0.0%)
Greatest absolute difference: 0.0004842388443648815 at index (7617, 2269) (up to 1e-05 allowed)
Greatest relative difference: 57.265357971191406 at index (7703, 2268) (up to 1.3e-06 allowed)
  FAIL [topk backward | softmax | tokens= 32768 | experts=2304 | topk=36 | pre_sm=True | grp_topk=0 | scale=1.0 | bias=False | dtype=torch.float32 | input=arange]
       grad max abs diff  : 4.842830e-04
       grad mean abs diff : 7.981259e-08
       Tensor-likes are not close!

Mismatched elements: 12442 / 75497472 (0.0%)
Greatest absolute difference: 0.00048428302397951484 at index (2090, 2270) (up to 1e-05 allowed)
Greatest relative difference: 57.26610565185547 at index (1878, 2268) (up to 1.3e-06 allowed)

Topk correctness: 4/6 passed


Benchmarking 6 topk config(s) (warmup=20, iters=100, dtype=torch.float32, pass=backward)...

  kernel   tokens experts topk score_fn pre_sm grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
---------------------------------------------------------------------------------------------------------------
    topk      128    2304   36  softmax   True      0  float32 backward    0.3828    0.4522    1.18x       334356
    topk      512    2304   36  softmax   True      0  float32 backward    0.4323    0.4272    0.99x      1184416
    topk     2048    2304   36  softmax   True      0  float32 backward    0.4728    0.4304    0.91x      4331331
    topk     8192    2304   36  softmax   True      0  float32 backward    0.5955    0.4625    0.78x     13756097
    topk    32768    2304   36  softmax   True      0  float32 backward    1.7476    0.8116    0.46x     18749854
    topk   131072    2304   36  softmax   True      0  float32 backward    6.0583    2.3994    0.40x     21634987
```
