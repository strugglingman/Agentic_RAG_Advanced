output "aws_account_id" {
  description = "AWS account ID where this stack is applied."
  value       = data.aws_caller_identity.current.account_id
}

output "vpc_id" {
  description = "VPC ID."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs."
  value       = local.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs."
  value       = local.private_subnet_ids
}

output "backup_bucket_name" {
  description = "Backup bucket name."
  value       = local.s3_backup_bucket_name
}

output "ecr_backend_repository_arn" {
  description = "Backend ECR repository ARN."
  value       = module.ecr.backend_repository_arn
}

output "ecr_frontend_repository_arn" {
  description = "Frontend ECR repository ARN."
  value       = module.ecr.frontend_repository_arn
}

output "eks_cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN."
  value       = module.eks.cluster_arn
}

output "eks_nodegroup_name" {
  description = "EKS node group name."
  value       = module.eks.nodegroup_name
}

output "rds_endpoint" {
  description = "RDS endpoint address."
  value       = module.rds.endpoint
}

output "elasticache_endpoint" {
  description = "ElastiCache endpoint."
  value       = module.elasticache.endpoint
}

output "acm_certificate_arn" {
  description = "ACM certificate ARN."
  value       = module.acm.certificate_arn
}

output "acm_dns_validation_records" {
  description = "DNS records required for ACM DNS validation."
  value       = module.acm.dns_validation_records
}

output "eks_access_entry_arns" {
  description = "EKS access entry ARNs keyed by logical entry name."
  value       = module.eks_access.access_entry_arns
}
