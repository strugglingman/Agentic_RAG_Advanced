variable "enabled" {
  description = "Whether to manage ECR repositories."
  type        = bool
  default     = false
}

variable "backend_repository_name" {
  description = "Backend ECR repository name."
  type        = string
}

variable "frontend_repository_name" {
  description = "Frontend ECR repository name."
  type        = string
}

variable "image_tag_mutability" {
  description = "Image tag mutability setting for repositories."
  type        = string
  default     = "MUTABLE"
}

variable "scan_on_push" {
  description = "Whether to enable image scanning on push."
  type        = bool
  default     = false
}

variable "encryption_type" {
  description = "Repository encryption type (AES256 or KMS)."
  type        = string
  default     = "AES256"
}

variable "kms_key" {
  description = "KMS key ARN when encryption_type is KMS."
  type        = string
  default     = null
  nullable    = true
}

variable "force_delete" {
  description = "Whether repositories can be force-deleted."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags for ECR repositories."
  type        = map(string)
  default     = {}
}
