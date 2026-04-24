variable "enabled" {
  description = "Whether to manage the RDS instance."
  type        = bool
  default     = false
}

variable "db_instance_identifier" {
  description = "RDS DB instance identifier."
  type        = string
}

variable "engine" {
  description = "Database engine."
  type        = string
}

variable "engine_version" {
  description = "Database engine version."
  type        = string
}

variable "instance_class" {
  description = "RDS instance class."
  type        = string
}

variable "db_name" {
  description = "Initial database name."
  type        = string
}

variable "username" {
  description = "Master username."
  type        = string
}

variable "password" {
  description = "Master password. Keep null for import-first unless rotation is planned."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "port" {
  description = "Database port."
  type        = number
  default     = 5432
}

variable "allocated_storage" {
  description = "Allocated storage in GiB."
  type        = number
}

variable "max_allocated_storage" {
  description = "Maximum allocated storage autoscaling limit in GiB."
  type        = number
}

variable "storage_type" {
  description = "Storage type."
  type        = string
}

variable "iops" {
  description = "Provisioned IOPS."
  type        = number
}

variable "storage_throughput" {
  description = "Storage throughput in MiB/s."
  type        = number
}

variable "storage_encrypted" {
  description = "Whether storage encryption is enabled."
  type        = bool
}

variable "kms_key_id" {
  description = "KMS key ARN used for storage encryption."
  type        = string
}

variable "db_subnet_group_name" {
  description = "DB subnet group name."
  type        = string
}

variable "vpc_security_group_ids" {
  description = "VPC security groups attached to the DB instance."
  type        = list(string)
}

variable "parameter_group_name" {
  description = "DB parameter group name."
  type        = string
}

variable "publicly_accessible" {
  description = "Whether the DB instance is publicly accessible."
  type        = bool
}

variable "multi_az" {
  description = "Whether to deploy Multi-AZ."
  type        = bool
}

variable "auto_minor_version_upgrade" {
  description = "Whether minor engine upgrades are applied automatically."
  type        = bool
}

variable "backup_retention_period" {
  description = "Backup retention period in days."
  type        = number
}

variable "backup_window" {
  description = "Preferred daily backup window (UTC)."
  type        = string
}

variable "maintenance_window" {
  description = "Preferred weekly maintenance window (UTC)."
  type        = string
}

variable "copy_tags_to_snapshot" {
  description = "Whether to copy tags to snapshots."
  type        = bool
}

variable "deletion_protection" {
  description = "Whether deletion protection is enabled."
  type        = bool
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds. 0 disables enhanced monitoring."
  type        = number
}

variable "performance_insights_enabled" {
  description = "Whether Performance Insights is enabled."
  type        = bool
}

variable "iam_database_authentication_enabled" {
  description = "Whether IAM database authentication is enabled."
  type        = bool
}

variable "ca_cert_identifier" {
  description = "RDS CA certificate identifier."
  type        = string
}

variable "network_type" {
  description = "Network type for the DB instance."
  type        = string
}

variable "apply_immediately" {
  description = "Whether to apply modifications immediately."
  type        = bool
  default     = false
}

variable "skip_final_snapshot" {
  description = "Whether to skip final snapshot on delete. Set to match imported baseline."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags for the RDS instance."
  type        = map(string)
  default     = {}
}
