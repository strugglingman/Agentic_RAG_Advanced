output "access_entry_arns" {
  description = "EKS access entry ARNs keyed by logical entry name."
  value       = { for key, entry in aws_eks_access_entry.this : key => entry.access_entry_arn }
}

output "access_entry_principals" {
  description = "EKS access entry principal ARNs keyed by logical entry name."
  value       = { for key, entry in aws_eks_access_entry.this : key => entry.principal_arn }
}
