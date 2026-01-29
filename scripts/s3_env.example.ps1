# Copy to scripts/s3_env.ps1 (ignored) or run directly in your session.
# Set your bucket and prefix once, then reuse the sync scripts.

$env:TFM_S3_BUCKET = "your-bucket-name"
$env:TFM_S3_PREFIX = "tfm-memoria"  # folder/prefix inside the bucket

Write-Host "TFM_S3_BUCKET=$env:TFM_S3_BUCKET" -ForegroundColor Green
Write-Host "TFM_S3_PREFIX=$env:TFM_S3_PREFIX" -ForegroundColor Green
