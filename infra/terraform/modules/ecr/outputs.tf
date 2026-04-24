output "backend_repository_arn" {
  description = "Backend ECR repository ARN."
  value       = try(aws_ecr_repository.backend[0].arn, null)
}

output "frontend_repository_arn" {
  description = "Frontend ECR repository ARN."
  value       = try(aws_ecr_repository.frontend[0].arn, null)
}
