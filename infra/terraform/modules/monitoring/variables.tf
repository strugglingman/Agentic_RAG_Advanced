variable "enabled" {
  description = "Whether to create monitoring resources."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Common tags."
  type        = map(string)
  default     = {}
}

variable "sns_topic_name" {
  description = "SNS topic name for alerts."
  type        = string
}

variable "alert_email_endpoint" {
  description = "Optional email endpoint to subscribe to SNS topic."
  type        = string
  default     = ""
}

variable "rds_instance_id" {
  description = "RDS DBInstanceIdentifier for CPU alarm."
  type        = string
}

variable "redis_alarm_dimensions" {
  description = "Dimension map for Redis alarm metric."
  type        = map(string)
}

variable "redis_engine_alarm_dimensions" {
  description = "Dimension map for Redis engine CPU alarm metric."
  type        = map(string)
}

variable "alarm_name_prefix" {
  description = "Prefix for legacy CloudWatch alarm names."
  type        = string
}

variable "eks_namespace" {
  description = "Kubernetes namespace used for EKS pod restart alarm."
  type        = string
  default     = "agentic-rag"
}
