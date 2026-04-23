variable "enabled" {
  description = "Whether to manage EKS cluster resources."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Common tags."
  type        = map(string)
  default     = {}
}

variable "cluster_tags" {
  description = "Optional tags for EKS cluster resource only."
  type        = map(string)
  default     = {}
}

variable "cluster_name" {
  description = "EKS cluster name."
  type        = string
}

variable "cluster_role_arn" {
  description = "IAM role ARN used by the EKS control plane."
  type        = string
}

variable "cluster_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.35"
}

variable "cluster_bootstrap_self_managed_addons" {
  description = "Whether EKS bootstraps self-managed addons at cluster creation time."
  type        = bool
  default     = false
}

variable "cluster_subnet_ids" {
  description = "Subnets used by EKS control plane ENIs."
  type        = list(string)
}

variable "cluster_endpoint_public_access" {
  description = "Whether EKS API endpoint is publicly accessible."
  type        = bool
  default     = true
}

variable "cluster_endpoint_private_access" {
  description = "Whether EKS API endpoint is privately accessible."
  type        = bool
  default     = true
}

variable "cluster_public_access_cidrs" {
  description = "CIDR blocks allowed to access the public EKS API endpoint."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cluster_authentication_mode" {
  description = "EKS cluster authentication mode."
  type        = string
  default     = "API"
}

variable "cluster_bootstrap_admin_permissions" {
  description = "Whether cluster creator gets bootstrap admin permissions."
  type        = bool
  default     = true
}

variable "cluster_ip_family" {
  description = "Kubernetes network IP family."
  type        = string
  default     = "ipv4"
}

variable "cluster_upgrade_support_type" {
  description = "EKS upgrade support type."
  type        = string
  default     = "STANDARD"
}

variable "cluster_zonal_shift_enabled" {
  description = "Whether EKS zonal shift is enabled."
  type        = bool
  default     = false
}

variable "cluster_enabled_log_types" {
  description = "Control-plane log types to enable."
  type        = list(string)
  default     = []
}

variable "nodegroup_enabled" {
  description = "Whether to manage the EKS managed node group."
  type        = bool
  default     = false
}

variable "nodegroup_name" {
  description = "Managed node group name."
  type        = string
  default     = "agentic-rag-general-ng"
}

variable "nodegroup_role_arn" {
  description = "IAM role ARN used by the managed node group."
  type        = string
  default     = ""
}

variable "nodegroup_subnet_ids" {
  description = "Subnets used by the managed node group."
  type        = list(string)
  default     = []
}

variable "nodegroup_capacity_type" {
  description = "Managed node group capacity type."
  type        = string
  default     = "ON_DEMAND"
}

variable "nodegroup_ami_type" {
  description = "Managed node group AMI type."
  type        = string
  default     = "AL2023_x86_64_STANDARD"
}

variable "nodegroup_instance_types" {
  description = "Managed node group instance types."
  type        = list(string)
  default     = ["t3.medium"]
}

variable "nodegroup_disk_size" {
  description = "Managed node group root volume size (GiB)."
  type        = number
  default     = 30
}

variable "nodegroup_min_size" {
  description = "Managed node group min size."
  type        = number
  default     = 0
}

variable "nodegroup_max_size" {
  description = "Managed node group max size."
  type        = number
  default     = 1
}

variable "nodegroup_desired_size" {
  description = "Managed node group desired size."
  type        = number
  default     = 0
}

variable "nodegroup_update_max_unavailable" {
  description = "Managed node group max unavailable nodes during update."
  type        = number
  default     = 1
}

variable "nodegroup_repair_enabled" {
  description = "Whether managed node group auto repair is enabled."
  type        = bool
  default     = true
}

variable "enable_ebs_csi_addon" {
  description = "Whether to manage aws-ebs-csi-driver addon."
  type        = bool
  default     = false
}

variable "ebs_csi_addon_version" {
  description = "Optional aws-ebs-csi-driver addon version."
  type        = string
  default     = null
}

variable "enable_cloudwatch_observability_addon" {
  description = "Whether to manage amazon-cloudwatch-observability addon."
  type        = bool
  default     = false
}

variable "cloudwatch_observability_addon_version" {
  description = "Optional amazon-cloudwatch-observability addon version."
  type        = string
  default     = null
}

variable "enable_ebs_csi_pod_identity_association" {
  description = "Whether to manage EBS CSI pod identity association."
  type        = bool
  default     = false
}

variable "ebs_csi_pod_identity_namespace" {
  description = "Namespace for EBS CSI pod identity association."
  type        = string
  default     = "kube-system"
}

variable "ebs_csi_pod_identity_service_account" {
  description = "Service account for EBS CSI pod identity association."
  type        = string
  default     = "ebs-csi-controller-sa"
}

variable "ebs_csi_pod_identity_role_arn" {
  description = "IAM role ARN for EBS CSI pod identity association."
  type        = string
  default     = ""
}

variable "enable_cloudwatch_pod_identity_association" {
  description = "Whether to manage CloudWatch pod identity association."
  type        = bool
  default     = false
}

variable "cloudwatch_pod_identity_namespace" {
  description = "Namespace for CloudWatch pod identity association."
  type        = string
  default     = "amazon-cloudwatch"
}

variable "cloudwatch_pod_identity_service_account" {
  description = "Service account for CloudWatch pod identity association."
  type        = string
  default     = "cloudwatch-agent"
}

variable "cloudwatch_pod_identity_role_arn" {
  description = "IAM role ARN for CloudWatch pod identity association."
  type        = string
  default     = ""
}
