param(
    [string]$ClusterName = "agentic-rag-eks",
    [string]$Region = "eu-north-1"
)

$ErrorActionPreference = "Stop"

Write-Host "Updating kubeconfig for EKS cluster '$ClusterName' in region '$Region'..."
aws eks update-kubeconfig --name $ClusterName --region $Region

Write-Host "Current kubectl context:"
kubectl config current-context
