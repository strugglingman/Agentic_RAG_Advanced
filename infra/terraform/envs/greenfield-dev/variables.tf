variable "aws_region" {
  description = "AWS region for this environment."
  type        = string
  default     = "eu-north-1"
}

variable "aws_profile" {
  description = "Optional shared AWS config profile name (for example sso-dev)."
  type        = string
  default     = ""
}

variable "project_name" {
  description = "Project name used in resource naming."
  type        = string
  default     = "agentic-rag"
}

variable "environment" {
  description = "Environment name."
  type        = string
  default     = "greenfield-dev"
}

variable "extra_tags" {
  description = "Additional tags merged onto resources."
  type        = map(string)
  default     = {}
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use (2 or 3)."
  type        = number
  default     = 3

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 3
    error_message = "az_count must be 2 or 3."
  }
}

variable "public_subnet_newbits" {
  description = "Newbits used to derive public subnet CIDRs from vpc_cidr."
  type        = number
  default     = 4
}

variable "private_subnet_newbits" {
  description = "Newbits used to derive private subnet CIDRs from vpc_cidr."
  type        = number
  default     = 4
}

variable "enable_nat_gateway" {
  description = "Whether to provision a NAT gateway for private subnet egress."
  type        = bool
  default     = true
}

variable "enable_security_groups" {
  description = "Whether to manage security groups via modules/security_groups."
  type        = bool
  default     = true
}

variable "enable_iam" {
  description = "Whether to create IAM roles/policies."
  type        = bool
  default     = true
}

variable "enable_ecr" {
  description = "Whether to create ECR repositories."
  type        = bool
  default     = true
}

variable "enable_s3_backup_bucket" {
  description = "Whether to create backup S3 bucket."
  type        = bool
  default     = true
}

variable "enable_eks" {
  description = "Whether to create EKS cluster."
  type        = bool
  default     = true
}

variable "enable_eks_nodegroup" {
  description = "Whether to create EKS managed node group."
  type        = bool
  default     = true
}

variable "enable_rds" {
  description = "Whether to create RDS instance."
  type        = bool
  default     = true
}

variable "enable_elasticache" {
  description = "Whether to create ElastiCache serverless cache."
  type        = bool
  default     = true
}

variable "enable_acm" {
  description = "Whether to request ACM certificate."
  type        = bool
  default     = false
}

variable "enable_monitoring" {
  description = "Whether to provision SNS + CloudWatch baseline alarms."
  type        = bool
  default     = false
}

variable "enable_eks_access_entries" {
  description = "Whether to manage EKS access entries from inputs below."
  type        = bool
  default     = true
}

variable "enable_legacy_ec2" {
  description = "Whether to create legacy single-EC2 runtime host."
  type        = bool
  default     = false
}

variable "s3_backup_bucket_name" {
  description = "Optional explicit backup bucket name. Empty means auto-generated from project/account/region."
  type        = string
  default     = ""
}

variable "s3_backup_force_destroy" {
  description = "Whether backup bucket can be force-destroyed."
  type        = bool
  default     = false
}

variable "s3_backup_sse_algorithm" {
  description = "SSE algorithm for backup bucket (AES256 or aws:kms)."
  type        = string
  default     = "AES256"
}

variable "s3_backup_kms_master_key_id" {
  description = "KMS key for backup bucket if SSE is aws:kms."
  type        = string
  default     = null
  nullable    = true
}

variable "s3_backup_lifecycle_expiration_days" {
  description = "Lifecycle expiration days for backup objects."
  type        = number
  default     = 30
}

variable "ecr_backend_repo_name" {
  description = "Optional explicit backend ECR repo name."
  type        = string
  default     = ""
}

variable "ecr_frontend_repo_name" {
  description = "Optional explicit frontend ECR repo name."
  type        = string
  default     = ""
}

variable "ecr_image_tag_mutability" {
  description = "ECR image tag mutability."
  type        = string
  default     = "IMMUTABLE"
}

variable "ecr_scan_on_push" {
  description = "Enable image vulnerability scan on push."
  type        = bool
  default     = true
}

variable "ecr_encryption_type" {
  description = "ECR encryption type (AES256 or KMS)."
  type        = string
  default     = "AES256"
}

variable "ecr_kms_key" {
  description = "KMS key ARN for ECR when encryption type is KMS."
  type        = string
  default     = null
  nullable    = true
}

variable "ecr_force_delete" {
  description = "Whether ECR repos can be force deleted."
  type        = bool
  default     = false
}

variable "eks_cluster_name" {
  description = "Optional explicit EKS cluster name."
  type        = string
  default     = ""
}

variable "eks_cluster_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.35"
}

variable "enable_eks_public_endpoint" {
  description = "Whether EKS API endpoint is publicly accessible."
  type        = bool
  default     = true
}

variable "enable_eks_private_endpoint" {
  description = "Whether EKS API endpoint is privately accessible."
  type        = bool
  default     = true
}

variable "eks_public_access_cidrs" {
  description = "Public CIDRs allowed to access the EKS API endpoint when enabled."
  type        = list(string)
  default     = []
}

variable "eks_cluster_enabled_log_types" {
  description = "Control-plane log types enabled for EKS."
  type        = list(string)
  default = [
    "api",
    "audit",
    "authenticator",
  ]
}

variable "eks_cluster_authentication_mode" {
  description = "EKS authentication mode."
  type        = string
  default     = "API"
}

variable "eks_cluster_bootstrap_admin_permissions" {
  description = "Whether cluster creator gets admin access by default."
  type        = bool
  default     = true
}

variable "eks_cluster_bootstrap_self_managed_addons" {
  description = "Whether EKS bootstraps self-managed add-ons at cluster creation."
  type        = bool
  default     = false
}

variable "eks_nodegroup_name" {
  description = "Suffix for node group name."
  type        = string
  default     = "general-ng"
}

variable "eks_nodegroup_instance_types" {
  description = "Managed node group instance types."
  type        = list(string)
  default = [
    "t3.medium",
  ]
}

variable "eks_nodegroup_disk_size" {
  description = "Managed node group disk size in GiB."
  type        = number
  default     = 40
}

variable "eks_nodegroup_min_size" {
  description = "Managed node group minimum size."
  type        = number
  default     = 1
}

variable "eks_nodegroup_max_size" {
  description = "Managed node group maximum size."
  type        = number
  default     = 3
}

variable "eks_nodegroup_desired_size" {
  description = "Managed node group desired size."
  type        = number
  default     = 1
}

variable "eks_nodegroup_capacity_type" {
  description = "Managed node group capacity type."
  type        = string
  default     = "ON_DEMAND"
}

variable "eks_nodegroup_ami_type" {
  description = "Managed node group AMI type."
  type        = string
  default     = "AL2023_x86_64_STANDARD"
}

variable "eks_nodegroup_update_max_unavailable" {
  description = "Max unavailable nodes during node group update."
  type        = number
  default     = 1
}

variable "enable_eks_ebs_csi_role" {
  description = "Whether to create IAM role for EBS CSI pod identity."
  type        = bool
  default     = true
}

variable "enable_eks_ebs_csi_addon" {
  description = "Whether to create EBS CSI addon."
  type        = bool
  default     = true
}

variable "eks_ebs_csi_addon_version" {
  description = "Optional EBS CSI addon version pin."
  type        = string
  default     = null
  nullable    = true
}

variable "enable_eks_ebs_csi_pod_identity_association" {
  description = "Whether to create EBS CSI pod identity association."
  type        = bool
  default     = true
}

variable "enable_eks_cloudwatch_observability_role" {
  description = "Whether to create IAM role for CloudWatch observability pod identity."
  type        = bool
  default     = false
}

variable "enable_eks_cloudwatch_observability_addon" {
  description = "Whether to create CloudWatch observability addon."
  type        = bool
  default     = false
}

variable "eks_cloudwatch_observability_addon_version" {
  description = "Optional CloudWatch observability addon version pin."
  type        = string
  default     = null
  nullable    = true
}

variable "enable_eks_cloudwatch_pod_identity_association" {
  description = "Whether to create CloudWatch pod identity association."
  type        = bool
  default     = false
}

variable "eks_admin_principal_arns" {
  description = "IAM principal ARNs to grant EKS cluster-admin policy via access entries."
  type        = list(string)
  default     = []
}

variable "additional_eks_access_entries" {
  description = "Additional custom EKS access entries."
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

variable "rds_instance_identifier" {
  description = "Optional explicit RDS instance identifier."
  type        = string
  default     = ""
}

variable "rds_engine" {
  description = "RDS engine."
  type        = string
  default     = "postgres"
}

variable "rds_engine_version" {
  description = "RDS engine version."
  type        = string
  default     = "15.17"
}

variable "rds_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "rds_db_name" {
  description = "RDS database name."
  type        = string
  default     = "chatbot"
}

variable "rds_master_username" {
  description = "RDS master username."
  type        = string
  default     = "postgres"
}

variable "rds_master_password" {
  description = "RDS master password. Must be set in local tfvars when enable_rds=true."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "rds_port" {
  description = "RDS port."
  type        = number
  default     = 5432
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage (GiB)."
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS max allocated storage (GiB)."
  type        = number
  default     = 200
}

variable "rds_storage_type" {
  description = "RDS storage type."
  type        = string
  default     = "gp3"
}

variable "rds_iops" {
  description = "RDS IOPS."
  type        = number
  default     = 3000
}

variable "rds_storage_throughput" {
  description = "RDS storage throughput (MiB/s)."
  type        = number
  default     = 125
}

variable "rds_kms_key_id" {
  description = "RDS KMS key ID/ARN."
  type        = string
  default     = "alias/aws/rds"
}

variable "rds_parameter_group_name" {
  description = "RDS parameter group name."
  type        = string
  default     = "default.postgres15"
}

variable "rds_multi_az" {
  description = "Whether RDS is Multi-AZ."
  type        = bool
  default     = false
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period in days."
  type        = number
  default     = 7
}

variable "rds_backup_window" {
  description = "Preferred RDS backup window."
  type        = string
  default     = "03:00-03:30"
}

variable "rds_maintenance_window" {
  description = "Preferred RDS maintenance window."
  type        = string
  default     = "sun:04:00-sun:04:30"
}

variable "rds_deletion_protection" {
  description = "Whether RDS deletion protection is enabled."
  type        = bool
  default     = true
}

variable "rds_monitoring_interval" {
  description = "RDS enhanced monitoring interval in seconds."
  type        = number
  default     = 0
}

variable "rds_performance_insights_enabled" {
  description = "Whether RDS Performance Insights is enabled."
  type        = bool
  default     = false
}

variable "rds_iam_database_authentication_enabled" {
  description = "Whether RDS IAM database auth is enabled."
  type        = bool
  default     = false
}

variable "rds_ca_cert_identifier" {
  description = "RDS CA certificate identifier."
  type        = string
  default     = "rds-ca-rsa2048-g1"
}

variable "rds_apply_immediately" {
  description = "Whether RDS modifications apply immediately."
  type        = bool
  default     = false
}

variable "rds_skip_final_snapshot" {
  description = "Whether to skip final snapshot on RDS deletion."
  type        = bool
  default     = false
}

variable "elasticache_serverless_cache_name" {
  description = "Optional explicit ElastiCache serverless cache name."
  type        = string
  default     = ""
}

variable "elasticache_engine" {
  description = "ElastiCache engine."
  type        = string
  default     = "valkey"
}

variable "elasticache_major_engine_version" {
  description = "ElastiCache major engine version."
  type        = string
  default     = "8"
}

variable "elasticache_description" {
  description = "ElastiCache description."
  type        = string
  default     = "Agentic RAG cache"
}

variable "elasticache_kms_key_id" {
  description = "Optional ElastiCache KMS key ID/ARN."
  type        = string
  default     = null
  nullable    = true
}

variable "elasticache_user_group_id" {
  description = "Optional ElastiCache user group ID."
  type        = string
  default     = null
  nullable    = true
}

variable "elasticache_snapshot_retention_limit" {
  description = "ElastiCache snapshot retention limit."
  type        = number
  default     = 1
}

variable "elasticache_daily_snapshot_time" {
  description = "ElastiCache daily snapshot time."
  type        = string
  default     = "03:30"
}

variable "acm_domain_name" {
  description = "Primary domain name for ACM certificate (required when enable_acm=true)."
  type        = string
  default     = ""
}

variable "acm_subject_alternative_names" {
  description = "ACM subject alternative names."
  type        = list(string)
  default     = []
}

variable "acm_validation_method" {
  description = "ACM validation method."
  type        = string
  default     = "DNS"
}

variable "acm_key_algorithm" {
  description = "ACM key algorithm."
  type        = string
  default     = "RSA_2048"
}

variable "acm_certificate_transparency_logging_preference" {
  description = "ACM certificate transparency logging preference."
  type        = string
  default     = "ENABLED"
}

variable "sns_topic_name" {
  description = "Optional explicit SNS topic name for alerts."
  type        = string
  default     = ""
}

variable "alert_email_endpoint" {
  description = "Optional email endpoint for SNS subscription."
  type        = string
  default     = ""
}

variable "eks_namespace" {
  description = "Kubernetes namespace for EKS pod restart alarm."
  type        = string
  default     = "agentic-rag"
}

variable "rds_instance_id_for_monitoring" {
  description = "Optional RDS instance ID override for alarms when RDS is external."
  type        = string
  default     = ""
}

variable "redis_alarm_dimensions_override" {
  description = "Optional override map for Redis alarm dimensions."
  type        = map(string)
  default     = {}
}

variable "redis_engine_alarm_dimensions_override" {
  description = "Optional override map for Redis engine CPU alarm dimensions."
  type        = map(string)
  default     = {}
}

variable "ec2_ami_id" {
  description = "AMI ID for optional legacy EC2 host."
  type        = string
  default     = ""
}

variable "ec2_instance_type" {
  description = "Instance type for optional legacy EC2 host."
  type        = string
  default     = "t3.small"
}

variable "ec2_key_name" {
  description = "SSH key name for optional legacy EC2 host."
  type        = string
  default     = ""
}

variable "ec2_ingress_cidrs" {
  description = "CIDRs allowed to reach optional legacy EC2 host (22/80/443)."
  type        = list(string)
  default     = []
}
