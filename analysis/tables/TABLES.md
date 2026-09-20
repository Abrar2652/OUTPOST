# Paper tables (auto-generated)

Regenerate: `python analysis/build_tables.py`. Every number is
recomputed from `analysis/runlogs/*.runlog`; do not hand-edit.

`best_*` = per-metric max over epochs (the protocol used by every
published baseline). `atbestval_*` = metrics at the peak-validation
epoch (honest selection). Both are reported everywhere.

Only tables computed from the CURRENT run record are rendered here. The runlog-derived tables (`table_yelp_multiseed`, `table_photo`, `table_compression`, `table_yelp_ablation`, `runs`) are the July three-seed archive and live in `analysis/tables/legacy/`.

## table_main_comparison

| Method      |   Photo_AUC-ROC |   Photo_AUC-PR |   Computers_AUC-ROC |   Computers_AUC-PR |   CS_AUC-ROC |   CS_AUC-PR |   Yelp_AUC-ROC |   Yelp_AUC-PR |   ogbn-arxiv_AUC-ROC |   ogbn-arxiv_AUC-PR |   ogbn-mag_AUC-ROC |   ogbn-mag_AUC-PR |
|:------------|----------------:|---------------:|--------------------:|-------------------:|-------------:|------------:|---------------:|--------------:|---------------------:|--------------------:|-------------------:|------------------:|
| ANOMALOUS   |          0.5574 |         0.0879 |              0.5737 |             0.1693 |       0.2997 |      0.1634 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| DOMINANT    |          0.4716 |         0.0837 |              0.545  |             0.1644 |       0.4029 |      0.1886 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| AnomalyDAE  |          0.4179 |         0.077  |              0.5658 |             0.1723 |       0.3978 |      0.1864 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| GAAN        |          0.4346 |         0.071  |              0.5595 |             0.1796 |       0.4646 |      0.2111 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| CoLA        |          0.5618 |         0.0989 |              0.4897 |             0.1472 |       0.4353 |      0.2029 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| CONAD       |          0.4763 |         0.0862 |              0.5445 |             0.1619 |       0.4028 |      0.1886 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| ConsisGAD   |          0.8668 |         0.5987 |              0.625  |             0.3572 |       0.7178 |      0.5271 |         0.6988 |        0.297  |               0.6216 |              0.3148 |             0.4909 |            0.0043 |
| GGAD        |          0.7976 |         0.5677 |              0.721  |             0.4529 |       0.9081 |      0.8198 |         0.6613 |        0.2549 |               0.6007 |              0.2843 |           nan      |          nan      |
| TAM         |          0.6045 |         0.1084 |              0.4432 |             0.1355 |       0.6398 |      0.3542 |         0.5319 |        0.0977 |             nan      |            nan      |           nan      |          nan      |
| OGCNN       |          0.6279 |         0.1323 |              0.5049 |             0.1505 |       0.7819 |      0.4926 |         0.641  |        0.1118 |             nan      |            nan      |           nan      |          nan      |
| ANO-S       |          0.573  |         0.1097 |              0.4628 |             0.1392 |       0.838  |      0.6401 |         0.6567 |        0.1076 |               0.451  |              0.1463 |           nan      |          nan      |
| DOM-S       |          0.5785 |         0.1107 |              0.4488 |             0.133  |       0.8445 |      0.6382 |         0.6506 |        0.1048 |               0.4505 |              0.1482 |           nan      |          nan      |
| SpaceGNN    |          0.803  |         0.5271 |              0.8296 |             0.6439 |       0.7784 |      0.6587 |         0.6853 |        0.2916 |               0.6133 |              0.3301 |             0.4626 |            0.0043 |
| NSReg       |          0.836  |         0.4777 |              0.7403 |             0.5437 |       0.9032 |      0.8115 |         0.7015 |        0.3029 |               0.6182 |              0.323  |             0.4836 |            0.0041 |
| GNN+OpenMax |          0.7618 |         0.3942 |              0.6713 |             0.3942 |       0.8213 |      0.7559 |       nan      |      nan      |             nan      |            nan      |           nan      |          nan      |
| DEMO        |          0.9023 |         0.633  |              0.8439 |             0.6458 |       0.9448 |      0.8857 |         0.7097 |        0.2238 |               0.6364 |              0.3329 |             0.4967 |            0.0054 |
| OUTPOST     |          0.8703 |         0.5557 |              0.851  |             0.6521 |       0.9842 |      0.9575 |         0.7448 |        0.3824 |               0.6229 |              0.3049 |             0.5928 |            0.0101 |

## table_complexity

| dataset   | model   |   hidden |   spectral_gate |   params |   ratio_vs_demo |
|:----------|:--------|---------:|----------------:|---------:|----------------:|
| photo     | DEMO    |       64 |             nan |   725819 |          1      |
| photo     | OUTPOST |       64 |               1 |   209730 |          0.289  |
| photo     | OUTPOST |       64 |               0 |   114113 |          0.1572 |
| photo     | OUTPOST |       32 |               1 |   148610 |          0.2047 |
| photo     | OUTPOST |       32 |               0 |    52993 |          0.073  |
| photo     | OUTPOST |       16 |               1 |   121122 |          0.1669 |
| photo     | OUTPOST |       16 |               0 |    25505 |          0.0351 |
| yelp      | DEMO    |       64 |             nan |    34209 |          1      |
| yelp      | OUTPOST |       64 |               1 |    27202 |          0.7952 |
| yelp      | OUTPOST |       64 |               0 |    22849 |          0.6679 |
| yelp      | OUTPOST |       32 |               1 |    11714 |          0.3424 |
| yelp      | OUTPOST |       32 |               0 |     7361 |          0.2152 |
| yelp      | OUTPOST |       16 |               1 |     7042 |          0.2059 |
| yelp      | OUTPOST |       16 |               0 |     2689 |          0.0786 |
