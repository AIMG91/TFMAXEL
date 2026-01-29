[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [string] $Bucket = $env:TFM_S3_BUCKET,

  [Parameter(Mandatory = $false)]
  [string] $Prefix = $env:TFM_S3_PREFIX,

  [Parameter(Mandatory = $false)]
  [string] $LocalOutputs = "outputs",

  [switch] $Delete
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Bucket)) {
  throw "S3 bucket not set. Provide -Bucket or set env var TFM_S3_BUCKET."
}
if ([string]::IsNullOrWhiteSpace($Prefix)) {
  throw "S3 prefix not set. Provide -Prefix or set env var TFM_S3_PREFIX (e.g. tfm-memoria)."
}
if (-not (Test-Path -LiteralPath $LocalOutputs)) {
  throw "Local outputs path not found: $LocalOutputs"
}

$normalizedPrefix = $Prefix.Trim('/')
$destination = "s3://$Bucket/$normalizedPrefix/outputs"

Write-Host "Syncing $LocalOutputs -> $destination" -ForegroundColor Cyan

$cmd = @('s3','sync', $LocalOutputs, $destination)
if ($Delete) { $cmd += '--delete' }

aws @cmd
