# Informe de experimentos (E1–E5)

Este resumen consolida los resultados generados en `outputs/metrics` y el CSV consolidado `outputs/experiments/summary_metrics.csv`.

## Modelos evaluados
- SARIMAX (`sarimax_exog`)
- Prophet (`prophet_regressors`)
- DeepAR variantes (lags/exógenas, distintos hidden sizes y dropout)
- LSTM global (`lstm_exog`)
- Transformer global (`transformer_exog`, varias configuraciones de `dm/nh/nl/dropout/lr`)

## Configuración común
- Horizonte de test: últimas 39 semanas
- Exógenas: Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- Lags: 1, 2, 4, 8, 52; Rollings: 4, 8, 12; calendario opcional
- Métricas: MAE, RMSE, sMAPE

## Experimentos (src/experiments.py)
- E1 Walk-forward (h=1, ventana rolling opcional)
- E2 Últimas 39 semanas (h=39)
- E3 LOSO (leave-one-store-out) para modelos globales
- E4 Train35/Test10Low (tiendas con ventas bajas en test)
- E5 Shock desempleo +15% (sensibilidad a exógena)

## Resultados globales (top-5)
| Rank | Modelo | MAE | RMSE | sMAPE |
| :-- | :-- | --: | --: | --: |
| 1 | transformer_exog<br>dm128 nh8 nl4 do0.1 lr0.0003 | 50989.6019 | 68451.7314 | 5.5283 |
| 2 | deepar_exog<br>hs80 nl2 do0.1 lr0.0005 bs32 | 58954.4875 | 82957.6729 | 6.1593 |
| 3 | deepar_exog (baseline) | 59573.6846 | 86287.6172 | 6.0413 |
| 4 | deepar_exog<br>hs40 nl2 do0.1 lr0.001 bs32 | 61163.8003 | 89421.2408 | 6.1288 |
| 5 | deepar_exog<br>hs80 nl3 do0.2 lr0.001 bs64 | 62260.5586 | 89886.1418 | 6.1915 |

## Tabla de métricas (top-10 global, MAE ascendente)
| Modelo | MAE | RMSE | sMAPE |
| :-- | --: | --: | --: |
| transformer_exog<br>dm128 nh8 nl4 do0.1 lr0.0003 | 50989.6019 | 68451.7314 | 5.5283 |
| deepar_exog<br>hs80 nl2 do0.1 lr0.0005 bs32 | 58954.4875 | 82957.6729 | 6.1593 |
| deepar_exog (baseline) | 59573.6846 | 86287.6172 | 6.0413 |
| deepar_exog<br>hs40 nl2 do0.1 lr0.001 bs32 | 61163.8003 | 89421.2408 | 6.1288 |
| deepar_exog<br>hs80 nl3 do0.2 lr0.001 bs64 | 62260.5586 | 89886.1418 | 6.1915 |
| deepar_exog<br>hs40 nl3 do0.2 lr0.0005 bs64 | 62739.1703 | 89809.3466 | 6.4222 |
| transformer_exog<br>dm64 nh4 nl2 do0.2 lr0.0003 | 64173.7257 | 85342.6788 | 6.6937 |
| transformer_exog<br>dm128 nh8 nl2 do0.2 lr0.001 | 82876.3535 | 104778.9802 | 11.0541 |
| transformer_exog<br>dm64 nh4 nl2 do0.1 lr0.001 | 87233.7185 | 105921.9935 | 10.9303 |
| sarimax_exog | 124157.6803 | 169022.4700 | 10.8893 |

## Resultados por tienda (media por modelo)
- MAE/RMSE: lidera transformer_exog__dm128__nh8__nl4__do0.1__lr0.0003, seguido por variantes DeepAR hs80_nl2_do0.1_lr0.0005_bs32 y baseline.
- sMAPE: mejor deepar_exog baseline, luego DeepAR hs40/hs80 y el transformer.

## Archivos clave
- Consolidados: outputs/experiments/summary_metrics.csv
- Métricas individuales: outputs/metrics/*_metrics_global.csv y *_metrics_by_store.csv
- Figuras: outputs/figures/

## Cómo generar/actualizar
1) Ejecutar en el notebook 10_Run_All_Experiments.ipynb:
   - Celda de “Resumen de métricas” (genera summary_metrics.csv desde outputs/metrics)
   - O la celda de agregación con `load(...)` si prefieres rehacer desde CSV locales.
2) Asegurar que `outputs/metrics` está presente en el entorno (SageMaker/local) antes de generar.

## Sugerencias
- Comparativa visual: cargar summary_metrics.csv en un notebook y graficar barplots de MAE/RMSE/sMAPE por modelo.
- Para shock E5: revisar outputs/experiments/E5/*_shock_summary.csv y figuras asociadas.

## Para exportar a PDF (rápido)
- Abrir este markdown en VS Code y “Export as PDF” (extensión Markdown PDF) o usar Pandoc:
  `pandoc outputs/experiments/report.md -o outputs/experiments/report.pdf`
