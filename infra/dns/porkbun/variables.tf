variable "aws_region" {
  description = "AWS region used by ACM certificate validation."
  type        = string
  default     = "eu-north-1"
}

variable "domain" {
  description = "Authoritative DNS zone in Porkbun."
  type        = string
}

variable "aws_state_bucket" {
  description = "S3 state bucket for infra/terraform env remote state."
  type        = string
  default     = "agentic-rag-tfstate-543035741679-eu-north-1"
}

variable "aws_state_key" {
  description = "State key for infra/terraform env that owns ACM resources."
  type        = string
  default     = "envs/dev/terraform.tfstate"
}

variable "aws_state_region" {
  description = "Region for the S3 backend that stores infra/terraform env state."
  type        = string
  default     = "eu-north-1"
}

variable "porkbun_api_key" {
  description = "Porkbun API key."
  type        = string
  sensitive   = true
}

variable "porkbun_secret_api_key" {
  description = "Porkbun Secret API key."
  type        = string
  sensitive   = true
}

variable "manage_acm_validation_records" {
  description = "Whether to manage ACM DNS validation records in Porkbun."
  type        = bool
  default     = true
}

variable "acm_validation_ttl" {
  description = "TTL for ACM validation CNAME records."
  type        = number
  default     = 600
}

variable "enable_acm_certificate_validation" {
  description = "Whether to wait for ACM certificate issuance after DNS records are managed."
  type        = bool
  default     = true
}

variable "app_dns_records" {
  description = "Application DNS records managed in Porkbun (for example www/api CNAME to ALB)."
  type = list(object({
    name     = string
    type     = optional(string, "CNAME")
    content  = string
    ttl      = optional(number, 600)
    priority = optional(number)
    notes    = optional(string)
  }))
  default = []
}
