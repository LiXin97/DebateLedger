# Thinking-off rerun: truncation report and sensitivities (rerun)

Truncated = finish_reason "length" at the 8192 cap. Main numbers keep the original parser.

## 1. Per row and stage: truncated calls / think tags / repetitive outputs

| row | stage | n | truncated | think tags | repetitive | answer None |
|---|---|---:|---:|---:|---:|---:|
| qwen3.5-4b | R0 | 600 | 43 (7.17%) | 27 | 3 | 0 |
| qwen3.5-4b | R1 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.5-4b | R2 | 600 | 1 (0.17%) | 2 | 0 | 0 |
| qwen3.5-4b | R3 | 600 | 1 (0.17%) | 0 | 0 | 0 |
| qwen3.5-4b | R4 | 600 | 1 (0.17%) | 0 | 0 | 0 |
| qwen3.5-4b | initial | 1800 | 127 (7.06%) | 101 | 2 | 0 |
| qwen3.5-4b | probe | 14400 | 557 (3.87%) | 243 | 17 | 3 |
| qwen3.5-9b | R0 | 600 | 32 (5.33%) | 24 | 1 | 0 |
| qwen3.5-9b | R1 | 600 | 1 (0.17%) | 0 | 0 | 0 |
| qwen3.5-9b | R2 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.5-9b | R3 | 600 | 1 (0.17%) | 0 | 0 | 0 |
| qwen3.5-9b | R4 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.5-9b | initial | 1800 | 118 (6.56%) | 104 | 1 | 0 |
| qwen3.5-9b | probe | 14400 | 525 (3.65%) | 172 | 19 | 0 |
| qwen3.6-27b-fp8 | R0 | 600 | 7 (1.17%) | 0 | 0 | 0 |
| qwen3.6-27b-fp8 | R1 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-27b-fp8 | R2 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-27b-fp8 | R3 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-27b-fp8 | R4 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-27b-fp8 | initial | 1800 | 31 (1.72%) | 0 | 1 | 0 |
| qwen3.6-27b-fp8 | probe | 14400 | 153 (1.06%) | 0 | 4 | 0 |
| qwen3.6-35b-a3b-fp8 | R0 | 600 | 9 (1.50%) | 1 | 0 | 0 |
| qwen3.6-35b-a3b-fp8 | R1 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-35b-a3b-fp8 | R2 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-35b-a3b-fp8 | R3 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-35b-a3b-fp8 | R4 | 600 | 0 (0.00%) | 0 | 0 | 0 |
| qwen3.6-35b-a3b-fp8 | initial | 1800 | 33 (1.83%) | 0 | 0 | 0 |
| qwen3.6-35b-a3b-fp8 | probe | 14400 | 100 (0.69%) | 0 | 1 | 0 |

## 2. alpha_tot: main vs truncation sensitivities

| row | triples | main (original parser) | drop truncated triples (kept) | strict final statement [undecided as no / as yes] | strict: undecided initial / replies |
|---|---:|---:|---:|---:|---:|
| qwen3.5-4b | 1800 | 0.3974 | 0.3424 (1431) | 0.3605 [0.3490, 0.3729] | 127 / 320 |
| qwen3.5-9b | 1800 | 0.2721 | 0.2204 (1486) | 0.2397 [0.2310, 0.2520] | 118 / 282 |
| qwen3.6-27b-fp8 | 1800 | 0.1160 | 0.0991 (1738) | 0.1027 [0.1012, 0.1058] | 32 / 65 |
| qwen3.6-35b-a3b-fp8 | 1800 | 0.1094 | 0.0950 (1749) | 0.0978 [0.0966, 0.0993] | 34 / 38 |

## 3. C^cond: main vs debates with truncation dropped

| row | debates | with truncation in R0-R3 | R3_alpha main | R3_alpha dropped | R4_stored main | R4_stored dropped |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.5-4b | 200 | 20 | 10/154 (6.49%) | 7/144 (4.86%) | 11/154 (7.14%) | 8/144 (5.56%) |
| qwen3.5-9b | 200 | 17 | 4/158 (2.53%) | 3/151 (1.99%) | 4/158 (2.53%) | 3/151 (1.99%) |
| qwen3.6-27b-fp8 | 200 | 5 | 4/166 (2.41%) | 2/162 (1.23%) | 4/166 (2.41%) | 2/162 (1.23%) |
| qwen3.6-35b-a3b-fp8 | 200 | 5 | 4/164 (2.44%) | 3/162 (1.85%) | 6/164 (3.66%) | 5/162 (3.09%) |

## 4. Truncated vs complete calls

| row | debate R0 correct: truncated / complete | probe initial correct: truncated / complete | probe reply revised: truncated / complete |
|---|---:|---:|---:|
| qwen3.5-4b | 19/43 (44.2%) / 435/557 (78.1%) | 28/127 (22.0%) / 1231/1673 (73.6%) | 429/557 (77.0%) / 5293/13843 (38.2%) |
| qwen3.5-9b | 8/32 (25.0%) / 463/568 (81.5%) | 32/118 (27.1%) / 1283/1682 (76.3%) | 359/525 (68.4%) / 3559/13875 (25.7%) |
| qwen3.6-27b-fp8 | 4/7 (57.1%) / 487/593 (82.1%) | 5/31 (16.1%) / 1471/1769 (83.2%) | 118/153 (77.1%) / 1553/14247 (10.9%) |
| qwen3.6-35b-a3b-fp8 | 2/9 (22.2%) / 485/591 (82.1%) | 8/33 (24.2%) / 1461/1767 (82.7%) | 75/100 (75.0%) / 1500/14300 (10.5%) |

## 5. Questions with >= 3 truncated calls (for provenance)

- qwen3.5-4b: mmlu_pro_8114 50; mmlu_pro_4097 40; mmlu_pro_3491 36; mmlu_pro_313 33; mmlu_pro_11926 32; mmlu_pro_7689 31; mmlu_pro_5766 27; mmlu_pro_342 25; mmlu_pro_1877 19; mmlu_pro_11151 18; mmlu_pro_11577 17; mmlu_pro_6250 15; mmlu_pro_9898 14; mmlu_pro_11636 13; mmlu_pro_438 13; mmlu_pro_4427 13; mmlu_pro_11195 11; mmlu_pro_11575 11; mmlu_pro_11998 11; mmlu_pro_914 11; mmlu_pro_1080 10; mmlu_pro_10970 9; mmlu_pro_1420 9; mmlu_pro_1272 8; mmlu_pro_1737 8; mmlu_pro_9161 8; mmlu_pro_3430 7; mmlu_pro_945 7; mmlu_pro_10332 6; mmlu_pro_11927 6; mmlu_pro_997 6; mmlu_pro_10234 5; mmlu_pro_1125 5; mmlu_pro_11584 5; mmlu_pro_4882 5; mmlu_pro_4998 5; mmlu_pro_909 5; mmlu_pro_9165 5; mmlu_pro_9984 5; mmlu_pro_11412 4; mmlu_pro_11918 4; mmlu_pro_1811 4; mmlu_pro_4010 4; mmlu_pro_4356 4; mmlu_pro_603 4; mmlu_pro_11150 3; mmlu_pro_1134 3; mmlu_pro_11399 3; mmlu_pro_11594 3; mmlu_pro_11600 3; mmlu_pro_11605 3; mmlu_pro_11671 3; mmlu_pro_11828 3; mmlu_pro_12015 3; mmlu_pro_1769 3; mmlu_pro_1784 3; mmlu_pro_1853 3; mmlu_pro_2232 3; mmlu_pro_4240 3; mmlu_pro_520 3; mmlu_pro_6312 3; mmlu_pro_6454 3; mmlu_pro_6678 3; mmlu_pro_6744 3; mmlu_pro_731 3; mmlu_pro_7315 3; mmlu_pro_8472 3
- qwen3.5-9b: mmlu_pro_8114 54; mmlu_pro_4097 51; mmlu_pro_6250 44; mmlu_pro_3491 43; mmlu_pro_11151 28; mmlu_pro_11926 28; mmlu_pro_7689 27; mmlu_pro_5766 21; mmlu_pro_342 18; mmlu_pro_11577 16; mmlu_pro_3525 15; mmlu_pro_4427 14; mmlu_pro_313 12; mmlu_pro_11195 11; mmlu_pro_10234 10; mmlu_pro_9984 10; mmlu_pro_9161 9; mmlu_pro_1053 8; mmlu_pro_10970 8; mmlu_pro_11412 8; mmlu_pro_11575 8; mmlu_pro_11927 8; mmlu_pro_1769 8; mmlu_pro_11584 7; mmlu_pro_4998 7; mmlu_pro_6312 7; mmlu_pro_914 7; mmlu_pro_1041 6; mmlu_pro_947 6; mmlu_pro_1080 5; mmlu_pro_1184 5; mmlu_pro_1819 5; mmlu_pro_4356 5; mmlu_pro_634 5; mmlu_pro_6454 5; mmlu_pro_6627 5; mmlu_pro_731 5; mmlu_pro_11594 4; mmlu_pro_11600 4; mmlu_pro_11636 4; mmlu_pro_1877 4; mmlu_pro_1984 4; mmlu_pro_9035 4; mmlu_pro_997 4; mmlu_pro_10768 3; mmlu_pro_11150 3; mmlu_pro_11605 3; mmlu_pro_11828 3; mmlu_pro_12015 3; mmlu_pro_1682 3; mmlu_pro_1784 3; mmlu_pro_1853 3; mmlu_pro_3438 3; mmlu_pro_356 3; mmlu_pro_4882 3; mmlu_pro_6446 3; mmlu_pro_9165 3; mmlu_pro_9898 3
- qwen3.6-27b-fp8: mmlu_pro_4097 41; mmlu_pro_8114 41; mmlu_pro_11151 27; mmlu_pro_3491 22; mmlu_pro_11577 20; mmlu_pro_11926 14; mmlu_pro_342 7; mmlu_pro_11195 5; mmlu_pro_7689 4
- qwen3.6-35b-a3b-fp8: mmlu_pro_4097 49; mmlu_pro_11577 21; mmlu_pro_3491 18; mmlu_pro_8114 14; mmlu_pro_11151 11; mmlu_pro_342 7; mmlu_pro_11926 6; mmlu_pro_11600 3
- in two or more rows: mmlu_pro_10234, mmlu_pro_1080, mmlu_pro_10970, mmlu_pro_11150, mmlu_pro_11151, mmlu_pro_11195, mmlu_pro_11412, mmlu_pro_11575, mmlu_pro_11577, mmlu_pro_11584, mmlu_pro_11594, mmlu_pro_11600, mmlu_pro_11605, mmlu_pro_11636, mmlu_pro_11828, mmlu_pro_11926, mmlu_pro_11927, mmlu_pro_12015, mmlu_pro_1769, mmlu_pro_1784, mmlu_pro_1853, mmlu_pro_1877, mmlu_pro_313, mmlu_pro_342, mmlu_pro_3491, mmlu_pro_4097, mmlu_pro_4356, mmlu_pro_4427, mmlu_pro_4882, mmlu_pro_4998, mmlu_pro_5766, mmlu_pro_6250, mmlu_pro_6312, mmlu_pro_6454, mmlu_pro_731, mmlu_pro_7689, mmlu_pro_8114, mmlu_pro_914, mmlu_pro_9161, mmlu_pro_9165, mmlu_pro_9898, mmlu_pro_997, mmlu_pro_9984
