# 第一次导出 + 跑视觉分支（CPU，自动流数，吞吐量优先，FP16）

numactl -C 0-15 python ov_test_p.py --mode demo --engine async --workers 2 --loops 100
