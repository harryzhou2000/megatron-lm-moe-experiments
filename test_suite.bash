set -e
set -u


python ../scripts/test_fused_topk.py --pass forward --csv ../data/naive_v2_topk_fwd${TEST_SUFFIX}.csv                          || true
python ../scripts/test_fused_topk.py --pass forward --csv ../data/naive_v2_aux_loss_fwd${TEST_SUFFIX}.csv --kernel aux_loss    || true

python ../scripts/test_fused_topk.py --pass backward --csv ../data/naive_v2_topk_bwd${TEST_SUFFIX}.csv                         || true
python ../scripts/test_fused_topk.py --pass backward --csv ../data/naive_v2_aux_loss_bwd${TEST_SUFFIX}.csv --kernel aux_loss   || true


