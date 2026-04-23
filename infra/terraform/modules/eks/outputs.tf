output "cluster_name" {
  description = "EKS cluster name."
  value       = try(aws_eks_cluster.this[0].name, null)
}

output "cluster_arn" {
  description = "EKS cluster ARN."
  value       = try(aws_eks_cluster.this[0].arn, null)
}

output "nodegroup_name" {
  description = "Managed node group name."
  value       = try(aws_eks_node_group.this[0].node_group_name, null)
}

output "nodegroup_arn" {
  description = "Managed node group ARN."
  value       = try(aws_eks_node_group.this[0].arn, null)
}

output "ebs_csi_addon_arn" {
  description = "aws-ebs-csi-driver addon ARN."
  value       = try(aws_eks_addon.ebs_csi[0].arn, null)
}

output "cloudwatch_observability_addon_arn" {
  description = "amazon-cloudwatch-observability addon ARN."
  value       = try(aws_eks_addon.cloudwatch_observability[0].arn, null)
}
