variable "aws_region" {
  description = "AWS region for bootstrap resources."
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Project name used in resource naming."
  type        = string
  default     = "agentic-rag"
}

variable "environment" {
  description = "Environment name (dev/staging/prod)."
  type        = string
  default     = "dev"
}
