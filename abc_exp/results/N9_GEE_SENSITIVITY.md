# N=9 GEE and GLM tables

4244 debates with a correct initial majority, 289 collapses, 9 models, 6 families.

| Fit | Term | Coef | SE | z | p |
|---|---|---:|---:|---:|---:|
| GEE logit, independence | alpha_adv | -0.760 | 0.774 | -0.98 | 0.326 |
| GEE logit, independence | init_acc | -4.204 | 2.295 | -1.83 | 0.067 |
| GEE logit, exchangeable | alpha_adv | +1.318 | 1.579 | +0.83 | 0.404 |
| GEE logit, exchangeable | init_acc | +0.198 | 1.997 | +0.10 | 0.921 |
| GLM logit, family-clustered SE | alpha_adv | -0.760 | 0.849 | -0.90 | 0.370 |
| GLM logit, family-clustered SE | init_acc | -4.204 | 2.515 | -1.67 | 0.095 |

Spearman between alpha_adv and initial accuracy: -0.70.

| Predictor of C^cond | Spearman rho | Fisher 90% CI |
|---|---:|---:|
| alpha_total | +0.78 | [+0.36, +0.94] |
| alpha_adv | +0.62 | [+0.05, +0.88] |
| init_acc | -0.73 | [-0.92, -0.26] |
