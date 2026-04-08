# fused_router performance optimization

perf script: [test_fused_topk.py](../scripts/test_fused_topk.py)

Reference is pytorch implementation (ref_ms).

There are 2 kernels in TE for fused router: topk and aux_loss.

Currently tested with fp32 fully.

Device: NVIDIA B300 SXM6 AC

## topk: forward

Before optimization: (main at 9d77dcb0638e7c3298c708df595035c0297cdad0)

``` log
  kernel   tokens experts topk     score_fn grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
------------------------------------------------------------------------------------------------------------
    topk      128     512   22      softmax      0  float32 forward    0.1701    0.0629    0.37x       752281
    topk      512     512   22      softmax      0  float32 forward    0.1701    0.0671    0.39x      3010416
    topk     2048     512   22      softmax      0  float32 forward    0.1803    0.1126    0.62x     11356659
    topk     8192     512   22      softmax      0  float32 forward    0.3912    0.2929    0.75x     20940712
    topk    32768     512   22      softmax      0  float32 forward    1.3045    0.8643    0.66x     25120006
    topk   131072     512   22      softmax      0  float32 forward    4.9508    3.6848    0.74x     26475118
aux_loss      128     512   22      softmax      0  float32 forward    0.1724    0.0381    0.22x       742639
aux_loss      512     512   22      softmax      0  float32 forward    0.1723    0.0474    0.28x      2971978
aux_loss     2048     512   22      softmax      0  float32 forward    0.1842    0.0905    0.49x     11119683
aux_loss     8192     512   22      softmax      0  float32 forward    0.3975    0.2697    0.68x     20608299
aux_loss    32768     512   22      softmax      0  float32 forward    1.2755    0.8245    0.65x     25691107
aux_loss   131072     512   22      softmax      0  float32 forward    4.8168    3.5569    0.74x     27211334
    topk      128     512   22      sigmoid      0  float32 forward    0.1703    0.0889    0.52x       751411
    topk      512     512   22      sigmoid      0  float32 forward    0.1722    0.0891    0.52x      2973745
    topk     2048     512   22      sigmoid      0  float32 forward    0.1824    0.1411    0.77x     11230592
    topk     8192     512   22      sigmoid      0  float32 forward    0.3939*   0.3257    0.83x     20799294
    topk    32768     512   22      sigmoid      0  float32 forward    1.3170    0.9576    0.73x     24881335
    topk   131072     512   22      sigmoid      0  float32 forward    5.0007    3.9973    0.80x     26210775
aux_loss      128     512   22      sigmoid      0  float32 forward    0.1723    0.0554    0.32x       742763
aux_loss      512     512   22      sigmoid      0  float32 forward    0.1732    0.0647    0.37x      2955421
aux_loss     2048     512   22      sigmoid      0  float32 forward    0.1844    0.1113    0.60x     11108796
aux_loss     8192     512   22      sigmoid      0  float32 forward    0.3973*   0.2922    0.74x     20621181
aux_loss    32768     512   22      sigmoid      0  float32 forward    1.2743    0.8906    0.70x     25713725
aux_loss   131072     512   22      sigmoid      0  float32 forward    4.8147    3.8078    0.79x     27223062

```

After optimization: (c685f5465109695024c29eebaf380fe06af96432) (pr #2821)

```log
  kernel   tokens experts topk     score_fn grp_tk    dtype    pass  fused_ms    ref_ms speedup        tok/s
------------------------------------------------------------------------------------------------------------
    topk      128     512   22      softmax      0  float32 forward    0.0294    0.0610    2.07x      4351089
    topk      512     512   22      softmax      0  float32 forward    0.0288    0.0666    2.31x     17747214
    topk     2048     512   22      softmax      0  float32 forward    0.0329    0.1126    3.42x     62222308
    topk     8192     512   22      softmax      0  float32 forward    0.1092    0.2925    2.68x     75001389
    topk    32768     512   22      softmax      0  float32 forward    0.3981    0.8641    2.17x     82301353
    topk   131072     512   22      softmax      0  float32 forward    1.5234    3.6841    2.42x     86041779
aux_loss      128     512   22      softmax      0  float32 forward    0.0248    0.0381    1.54x      5156433
aux_loss      512     512   22      softmax      0  float32 forward    0.0247    0.0476    1.93x     20698041
aux_loss     2048     512   22      softmax      0  float32 forward    0.0329    0.0895    2.72x     62251966
aux_loss     8192     512   22      softmax      0  float32 forward    0.1077    0.2699    2.51x     76093551
aux_loss    32768     512   22      softmax      0  float32 forward    0.3911    0.8246    2.11x     83773951
aux_loss   131072     512   22      softmax      0  float32 forward    1.4996    3.5567    2.37x     87406585
    topk      128     512   22      sigmoid      0  float32 forward    0.0309    0.0883    2.86x      4139972
    topk      512     512   22      sigmoid      0  float32 forward    0.0309    0.0887    2.87x     16571037
    topk     2048     512   22      sigmoid      0  float32 forward    0.0350    0.1413    4.04x     58527662
    topk     8192     512   22      sigmoid      0  float32 forward    0.1184*   0.3253    2.75x     69188252
    topk    32768     512   22      sigmoid      0  float32 forward    0.4373    0.9576    2.19x     74930378
    topk   131072     512   22      sigmoid      0  float32 forward    1.6769    3.9932    2.38x     78163646
aux_loss      128     512   22      sigmoid      0  float32 forward    0.0268    0.0550    2.05x      4771163
aux_loss      512     512   22      sigmoid      0  float32 forward    0.0267    0.0641    2.40x     19146542
aux_loss     2048     512   22      sigmoid      0  float32 forward    0.0332    0.1121    3.38x     61761158
aux_loss     8192     512   22      sigmoid      0  float32 forward    0.1170*   0.2920    2.50x     70019778
aux_loss    32768     512   22      sigmoid      0  float32 forward    0.4265    0.8906    2.09x     76823756
aux_loss   131072     512   22      sigmoid      0  float32 forward    1.6510    3.8071    2.31x     79391322
```
