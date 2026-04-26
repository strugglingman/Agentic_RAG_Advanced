variable "enabled" {
  description = "Whether to manage EKS access entries and policy associations."
  type        = bool
  default     = false
}

variable "cluster_name" {
  description = "EKS cluster name."
  type        = string
}

variable "access_entries" {
  description = "EKS access entries keyed by logical name."
  type = map(object({
    principal_arn     = string
    type              = optional(string, "STANDARD")
    kubernetes_groups = optional(list(string), [])
    username          = optional(string, null)
    tags              = optional(map(string), {})
    policy_associations = optional(list(object({
      policy_arn        = string
      access_scope_type = optional(string, "cluster")
      namespaces        = optional(list(string), [])
    })), [])
  }))
  default = {}
}
