output "role_arns" {
  description = "IAM role ARNs keyed by logical role name."
  value       = { for key, role in aws_iam_role.this : key => role.arn }
}

output "eks_cluster_role_arn" {
  description = "EKS control plane role ARN."
  value       = try(aws_iam_role.this["eks_cluster"].arn, null)
}

output "eks_node_role_arn" {
  description = "EKS managed node group role ARN."
  value       = try(aws_iam_role.this["eks_node"].arn, null)
}

output "eks_ebs_csi_role_arn" {
  description = "EBS CSI Pod Identity role ARN."
  value       = try(aws_iam_role.this["eks_ebs_csi"].arn, null)
}

output "eks_cloudwatch_observability_role_arn" {
  description = "CloudWatch Observability Pod Identity role ARN."
  value       = try(aws_iam_role.this["eks_cloudwatch_observability"].arn, null)
}

output "ec2_backup_role_arn" {
  description = "Legacy EC2 backup/deploy role ARN."
  value       = try(aws_iam_role.this["ec2_backup"].arn, null)
}
