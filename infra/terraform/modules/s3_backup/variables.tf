variable "enabled" {
  description = "Whether to manage backup S3 bucket resources."
  type        = bool
  default     = false
}

variable "bucket_name" {
  description = "Backup S3 bucket name."
  type        = string
}

variable "force_destroy" {
  description = "Whether bucket can be force-destroyed."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags for backup bucket."
  type        = map(string)
  default     = {}
}

variable "manage_public_access_block" {
  description = "Whether to manage bucket public access block settings."
  type        = bool
  default     = true
}

variable "block_public_acls" {
  description = "Block new public ACLs."
  type        = bool
  default     = true
}

variable "ignore_public_acls" {
  description = "Ignore all public ACLs."
  type        = bool
  default     = true
}

variable "block_public_policy" {
  description = "Block public bucket policies."
  type        = bool
  default     = true
}

variable "restrict_public_buckets" {
  description = "Restrict public bucket policies."
  type        = bool
  default     = true
}

variable "manage_server_side_encryption" {
  description = "Whether to manage bucket server-side encryption."
  type        = bool
  default     = true
}

variable "bucket_key_enabled" {
  description = "Whether to enable bucket key for SSE."
  type        = bool
  default     = true
}

variable "sse_algorithm" {
  description = "SSE algorithm (AES256 or aws:kms)."
  type        = string
  default     = "AES256"
}

variable "kms_master_key_id" {
  description = "KMS key ARN for SSE-KMS."
  type        = string
  default     = null
  nullable    = true
}

variable "manage_ownership_controls" {
  description = "Whether to manage S3 object ownership controls."
  type        = bool
  default     = true
}

variable "object_ownership" {
  description = "Object ownership setting."
  type        = string
  default     = "BucketOwnerEnforced"
}

variable "manage_lifecycle_configuration" {
  description = "Whether to manage bucket lifecycle configuration."
  type        = bool
  default     = true
}

variable "transition_default_minimum_object_size" {
  description = "Transition minimum object size default behavior."
  type        = string
  default     = "all_storage_classes_128K"
}

variable "lifecycle_rule_id" {
  description = "Lifecycle rule ID."
  type        = string
  default     = "expire-postgres-backups-after-30-days"
}

variable "lifecycle_prefix" {
  description = "Lifecycle prefix filter."
  type        = string
  default     = "postgres/"
}

variable "lifecycle_expiration_days" {
  description = "Lifecycle expiration days."
  type        = number
  default     = 30
}
