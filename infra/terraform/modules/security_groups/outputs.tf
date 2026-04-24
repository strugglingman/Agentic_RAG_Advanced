output "ec2_security_group_id" {
  description = "EC2 security group ID."
  value       = try(aws_security_group.ec2[0].id, null)
}

output "rds_security_group_id" {
  description = "RDS security group ID."
  value       = try(aws_security_group.rds[0].id, null)
}

output "elasticache_security_group_id" {
  description = "ElastiCache security group ID."
  value       = try(aws_security_group.elasticache[0].id, null)
}

output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID."
  value       = try(aws_security_group.eks_cluster[0].id, null)
}

output "default_vpc_security_group_id" {
  description = "Default VPC security group ID."
  value       = try(aws_default_security_group.default_vpc[0].id, null)
}
