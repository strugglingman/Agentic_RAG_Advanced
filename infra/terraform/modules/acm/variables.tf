variable "enabled" {
  description = "Whether to manage ACM certificate resources."
  type        = bool
  default     = false
}

variable "domain_name" {
  description = "Primary domain name for ACM certificate."
  type        = string
}

variable "subject_alternative_names" {
  description = "Subject alternative names for ACM certificate."
  type        = list(string)
  default     = []
}

variable "validation_method" {
  description = "Validation method for ACM certificate."
  type        = string
  default     = "DNS"
}

variable "key_algorithm" {
  description = "Public key algorithm for ACM certificate."
  type        = string
  default     = "RSA_2048"
}

variable "certificate_transparency_logging_preference" {
  description = "Certificate transparency logging preference."
  type        = string
  default     = "ENABLED"
}

variable "tags" {
  description = "Tags for ACM certificate."
  type        = map(string)
  default     = {}
}
