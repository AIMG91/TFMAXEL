# TFM – Forecasting de ventas semanales (Walmart) con variables exógenas

Objetivo: predecir `Weekly_Sales` semanal por `Store` usando **únicamente modelos que incorporen covariables exógenas** y comparar enfoques clásicos vs IA.

**Supuesto experimental (oracle exog):** se asume disponibilidad de TODAS las covariables exógenas durante el horizonte de predicción. Debe declararse en notebooks y memoria.

## Estructura

- `data/` – dataset local (copia de `Walmart_Sales.csv`)
- `notebooks/` – un notebook por modelo + auditoría inicial
- `outputs/`
  - `predictions/` – predicciones estandarizadas por modelo
  - `metrics/` – métricas globales y por tienda
  - `figures/` – figuras para la memoria
- `src/` – utilidades comunes (`src/common.py`)

## Notebooks

- `00_setup_and_gpu_check.ipynb`
- `01_Setup_and_Data_Audit.ipynb` – validación, EDA mínima, split temporal fijo, metadata
- `02_data_and_feature_sets.ipynb`
- `03_SARIMAX_exog.ipynb`
- `04_Prophet_regressors.ipynb`
- `05_GluonTS_DeepAR_exog.ipynb`
- `06_LSTM_global_exog.ipynb`
- `07_Transformer_global_exog.ipynb`
- `08_run_E0_ablation_training.ipynb`
- `09_results_summary_and_plots.ipynb`
- `10_Run_All_Experiments.ipynb`

## Protocolo común

- Split temporal sin leakage (mismos cortes para todos los modelos)
- Features:
  - Lags del target: `lag_1, lag_2, lag_4, lag_8, lag_52`
  - Rolling stats del target (solo pasado): `roll_mean_4, roll_mean_8, roll_std_8`
  - Exógenas: `Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment`
  - (Opcional) calendario: `weekofyear, month, year`
- Métricas: MAE, RMSE, sMAPE
- Outputs estándar por modelo en `outputs/`

## Flujo recomendado (VS Code local + ejecución en SageMaker + artefactos en S3)

Objetivo: editar el código en tu PC (VS Code) y ejecutar en SageMaker (GPU), guardando resultados en S3 para bajarlos luego a local.

### 1) Una vez: preparar S3

- Crea un bucket S3 (ej.: `tfm-memoria-outputs-<tu-nombre>`)
- Define un prefijo para este proyecto dentro del bucket (ej.: `tfm-memoria`)

### 2) Una vez: preparar tu repo

- El repo incluye `.gitignore` para evitar subir artefactos pesados a Git.
- Scripts de sync en `scripts/`:
  - `scripts/s3_sync_to.ps1` (sube `outputs/` a S3)
  - `scripts/s3_sync_from.ps1` (baja `outputs/` desde S3)
  - `scripts/s3_env.example.ps1` (plantilla de variables de entorno)

### 3) En tu PC (VS Code): desarrollar

- Edita y commitea cambios normalmente:
  - `git add -A`
  - `git commit -m "..."`
  - `git push`

### 4) En SageMaker Studio: ejecutar con GPU

- Abre SageMaker Studio (idealmente con instancia con GPU)
- Clona el repo una vez, y luego actualiza con:
  - `git pull`
- Instala dependencias según tu entorno (venv/conda/kernel) y ejecuta notebooks/scripts.
- Verifica GPU (si aplica): `nvidia-smi`

### 5) Subir outputs a S3 (desde SageMaker)

- Configura bucket/prefijo (en la sesión actual):
  - `. ./scripts/s3_env.example.ps1`  (edita valores antes, o crea `scripts/s3_env.ps1`)
- Sube resultados:
  - `./scripts/s3_sync_to.ps1 -Delete`

### 6) Bajar outputs a tu PC (desde VS Code local)

- En PowerShell local, configura bucket/prefijo y baja:
  - `. ./scripts/s3_env.example.ps1`
  - `./scripts/s3_sync_from.ps1`
