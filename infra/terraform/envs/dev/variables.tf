variable "aws_region" {
  description = "AWS region for dev environment."
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Project name."
  type        = string
  default     = "agentic-rag"
}

variable "environment" {
  description = "Environment name."
  type        = string
  default     = "dev"
}

variable "enable_monitoring" {
  description = "Whether to manage monitoring resources in this environment."
  type        = bool
  default     = false
}

variable "sns_topic_name" {
  description = "SNS topic name for infrastructure alerts."
  type        = string
  default     = "agentic-rag-alerts"
}

variable "alert_email_endpoint" {
  description = "Optional email endpoint for SNS subscription."
  type        = string
  default     = ""
}

variable "rds_instance_identifier" {
  description = "Existing RDS DBInstanceIdentifier."
  type        = string
  default     = "agentic-rag-postgres"
}

variable "redis_alarm_dimensions" {
  description = "Dimensions map for Redis alarm metric in your environment."
  type        = map(string)
  default = {
    CacheClusterId = "agentic-rag-redis"
  }
}
