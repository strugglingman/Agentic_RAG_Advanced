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
