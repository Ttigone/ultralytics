
### sky (frames=596, targets=596)

| method | SNR med | SNR p25 | gain<0.8 | AUC | d' | score med | thr@FA1 | rec@FA=1 | FA/f | FA@rec.9 | FA@rec.5 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | - | - | 0% | 0.978 | 2.84 | 8.4 | 14.6 | 0.000 | 1 | 2135 | 69 | 3.1 | 3.4 |
| mrf_gate_ms | 1.05x | 0.89x | 21% | 0.971 | 2.63 | 8.3 | 15.7 | 0.000 | 1 | 2133 | 87 | 39.4 | 41.7 |

### urban (frames=511, targets=511)

| method | SNR med | SNR p25 | gain<0.8 | AUC | d' | score med | thr@FA1 | rec@FA=1 | FA/f | FA@rec.9 | FA@rec.5 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | - | - | 0% | 0.980 | 2.78 | 7.5 | 14.6 | 0.000 | 1 | 2457 | 139 | 3.1 | 3.4 |
| mrf_gate_ms | 1.06x | 1.02x | 1% | 0.973 | 2.61 | 7.3 | 14.6 | 0.000 | 1 | 2498 | 146 | 38.4 | 41.5 |

### ground (frames=753, targets=753)

| method | SNR med | SNR p25 | gain<0.8 | AUC | d' | score med | thr@FA1 | rec@FA=1 | FA/f | FA@rec.9 | FA@rec.5 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | - | - | 0% | 0.962 | 2.34 | 7.1 | 12.7 | 0.000 | 1 | 6958 | 106 | 3.1 | 3.5 |
| mrf_gate_ms | 1.06x | 1.00x | 10% | 0.950 | 2.14 | 6.9 | 12.7 | 0.004 | 2 | 6610 | 178 | 38.7 | 41.2 |

### SNR gain by target size (all classes)

| method | s<=6 (n=66) | s7-9 (n=918) | s10-12 (n=876) |
|---|---|---|---|
| raw | - | - | - |
| mrf_gate_ms | 1.03x | 1.05x | 1.07x |