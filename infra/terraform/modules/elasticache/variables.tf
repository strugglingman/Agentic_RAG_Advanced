variable "enabled" {
  description = "Whether to manage the ElastiCache serverless cache."
  type        = bool
  default     = false
}

variable "serverless_cache_name" {
  description = "ElastiCache serverless cache name."
  type        = string
}

variable "engine" {
  description = "Cache engine, for example redis or valkey."
  type        = string
}

variable "major_engine_version" {
  description = "Major engine version."
  type        = string
  default     = null
  nullable    = true
}

variable "description" {
  description = "Description of the serverless cache."
  type        = string
  default     = null
  nullable    = true
}

variable "kms_key_id" {
  description = "KMS key ARN for encryption at rest."
  type        = string
  default     = null
  nullable    = true
}

variable "user_group_id" {
  description = "Optional user group ID for ACL."
  type        = string
  default     = null
  nullable    = true
}

variable "security_group_ids" {
  description = "Security groups attached to the serverless cache."
  type        = list(string)
  default     = []
}

variable "subnet_ids" {
  description = "Subnet IDs for the serverless cache."
  type        = list(string)
  default     = []
}

variable "snapshot_retention_limit" {
  description = "Snapshot retention in days."
  type        = number
  default     = null
  nullable    = true
}

variable "daily_snapshot_time" {
  description = "Daily snapshot time in UTC (HH:MM)."
  type        = string
  default     = null
  nullable    = true
}

variable "tags" {
  description = "Tags for the serverless cache."
  type        = map(string)
  default     = {}
}
