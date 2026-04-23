variable "aws_region" {
  description = "AWS region for dev environment."
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Project name."
  type        = string
  default     = "agentic-rag"
}

variable "environment" {
  description = "Environment name."
  type        = string
  default     = "dev"
}

variable "enable_monitoring" {
  description = "Whether to manage monitoring resources in this environment."
  type        = bool
  default     = false
}

variable "sns_topic_name" {
  description = "SNS topic name for infrastructure alerts."
  type        = string
  default     = "agentic-rag-alerts"
}

variable "alert_email_endpoint" {
  description = "Optional email endpoint for SNS subscription."
  type        = string
  default     = ""
}

variable "rds_instance_identifier" {
  description = "Existing RDS DBInstanceIdentifier."
  type        = string
  default     = "agentic-rag-postgres"
}

variable "redis_alarm_dimensions" {
  description = "Dimensions map for Redis alarm metric in your environment."
  type        = map(string)
  default = {
    CacheClusterId = "agentic-rag-redis"
  }
}

variable "redis_engine_alarm_dimensions" {
  description = "Dimensions map for Redis engine CPU alarm metric in your environment."
  type        = map(string)
  default = {
    CacheName = "agentic-rag-redis"
  }
}

variable "eks_namespace" {
  description = "Kubernetes namespace used for pod restart alarm."
  type        = string
  default     = "agentic-rag"
}

variable "enable_iam" {
  description = "Whether to manage IAM resources in this environment. Keep false until import is planned."
  type        = bool
  default     = false
}

variable "aws_account_id" {
  description = "AWS account ID used for account-scoped IAM policy ARNs."
  type        = string
  default     = "543035741679"
}

variable "iam_tags" {
  description = "Tags for IAM resources. Defaults to empty to preserve imported IAM no-op plans; add tags as a separate governance change."
  type        = map(string)
  default     = {}
}

variable "eks_cluster_role_name" {
  description = "IAM role name for EKS control plane."
  type        = string
  default     = "agentic-rag-eks-cluster-role"
}

variable "eks_node_role_name" {
  description = "IAM role name for EKS managed node group EC2 workers."
  type        = string
  default     = "agentic-rag-eks-node-role"
}

variable "eks_ebs_csi_role_name" {
  description = "IAM role name for EBS CSI driver EKS Pod Identity."
  type        = string
  default     = "AmazonEKSPodIdentityAmazonEBSCSIDriverRole"
}

variable "eks_cloudwatch_observability_role_name" {
  description = "IAM role name for CloudWatch Observability EKS Pod Identity."
  type        = string
  default     = "agentic-rag-eks-cloudwatch-observability-role"
}

variable "enable_eks_ebs_csi_role" {
  description = "Whether to manage the existing EBS CSI Pod Identity role."
  type        = bool
  default     = true
}

variable "enable_eks_cloudwatch_observability_role" {
  description = "Whether to manage a CloudWatch Observability Pod Identity role. Current AWS inventory has no such role, so default is false."
  type        = bool
  default     = false
}

variable "ec2_backup_role_name" {
  description = "Legacy EC2 role name for S3 backup and ECR push operations."
  type        = string
  default     = "agentic-rag-ec2-s3-backup-role"
}

variable "ec2_backup_role_description" {
  description = "Existing description for the legacy EC2 backup/deploy role."
  type        = string
  default     = "Allow the Agentic RAG EC2 instance to upload, download, list, and delete PostgreSQL backup files in the backup S3 bucket."
}

variable "ec2_instance_profile_name" {
  description = "Instance profile name for the legacy EC2 role."
  type        = string
  default     = "agentic-rag-ec2-s3-backup-role"
}

variable "s3_backup_policy_name" {
  description = "Customer managed policy name for PostgreSQL backup bucket access."
  type        = string
  default     = "agentic-rag-s3-backup-policy"
}

variable "s3_backup_policy_description" {
  description = "Existing description for the PostgreSQL backup bucket access policy."
  type        = string
  default     = "Allow EC2 to list, upload, download, and delete PostgreSQL backup files in the agentic-rag backup bucket."
}

variable "ecr_push_policy_name" {
  description = "Customer managed policy name for pushing backend/frontend images to ECR."
  type        = string
  default     = "agentic-rag-ecr-push-policy"
}

variable "ecr_push_policy_description" {
  description = "Existing description for the ECR push policy."
  type        = string
  default     = "Allow the EC2 deployment host to authenticate to ECR and push frontend and backend images for the Agentic RAG project."
}

variable "s3_backup_bucket_name" {
  description = "S3 bucket used for PostgreSQL backup objects."
  type        = string
  default     = "agentic-rag-adv-s3-543035741679-eu-north-1-an"
}

variable "ecr_backend_repo_name" {
  description = "Backend ECR repository name."
  type        = string
  default     = "agentic-rag-backend"
}

variable "ecr_frontend_repo_name" {
  description = "Frontend ECR repository name."
  type        = string
  default     = "agentic-rag-frontend"
}

variable "enable_eks" {
  description = "Whether to manage EKS resources with Terraform."
  type        = bool
  default     = false
}

variable "eks_cluster_name" {
  description = "EKS cluster name."
  type        = string
  default     = "agentic-rag-eks"
}

variable "eks_cluster_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.35"
}

variable "eks_tags" {
  description = "Common tags for EKS nodegroup/addon/pod-identity resources. Default empty for no-op adoption."
  type        = map(string)
  default     = {}
}

variable "eks_cluster_tags" {
  description = "Tags for EKS cluster resource. Keep existing legacy tags during adoption."
  type        = map(string)
  default = {
    "alpha.eksctl.io/cluster-oidc-enabled" = "true"
  }
}

variable "eks_cluster_bootstrap_self_managed_addons" {
  description = "Whether EKS cluster bootstraps self-managed addons."
  type        = bool
  default     = false
}

variable "eks_cluster_role_arn" {
  description = "EKS control plane role ARN. Leave empty to reuse IAM module output."
  type        = string
  default     = "arn:aws:iam::543035741679:role/agentic-rag-eks-cluster-role"
}

variable "eks_cluster_subnet_ids" {
  description = "Subnet IDs used by the EKS cluster."
  type        = list(string)
  default = [
    "subnet-0f9fa5b7cd7170e81",
    "subnet-05159408d4ed93c97",
    "subnet-078fedfabcf3ff892",
  ]
}

variable "eks_cluster_endpoint_public_access" {
  description = "Whether the EKS API endpoint is publicly accessible."
  type        = bool
  default     = true
}

variable "eks_cluster_endpoint_private_access" {
  description = "Whether the EKS API endpoint is privately accessible."
  type        = bool
  default     = true
}

variable "eks_cluster_public_access_cidrs" {
  description = "Public CIDR allow-list for EKS API endpoint."
  type        = list(string)
  default = [
    "0.0.0.0/0",
  ]
}

variable "eks_cluster_authentication_mode" {
  description = "EKS cluster authentication mode."
  type        = string
  default     = "API"
}

variable "eks_cluster_bootstrap_admin_permissions" {
  description = "Whether creator bootstrap admin permissions are enabled."
  type        = bool
  default     = true
}

variable "eks_cluster_ip_family" {
  description = "EKS cluster IP family."
  type        = string
  default     = "ipv4"
}

variable "eks_cluster_upgrade_support_type" {
  description = "EKS upgrade support type."
  type        = string
  default     = "STANDARD"
}

variable "eks_cluster_zonal_shift_enabled" {
  description = "Whether EKS zonal shift is enabled."
  type        = bool
  default     = false
}

variable "eks_cluster_enabled_log_types" {
  description = "Control-plane logs to enable. Empty means disabled."
  type        = list(string)
  default     = []
}

variable "enable_eks_nodegroup" {
  description = "Whether to manage the EKS managed node group."
  type        = bool
  default     = false
}

variable "eks_nodegroup_name" {
  description = "Managed node group name."
  type        = string
  default     = "agentic-rag-general-ng"
}

variable "eks_node_role_arn" {
  description = "EKS managed node group role ARN. Leave empty to reuse IAM module output."
  type        = string
  default     = "arn:aws:iam::543035741679:role/agentic-rag-eks-node-role"
}

variable "eks_nodegroup_subnet_ids" {
  description = "Subnets used by EKS node group."
  type        = list(string)
  default = [
    "subnet-0f9fa5b7cd7170e81",
    "subnet-05159408d4ed93c97",
    "subnet-078fedfabcf3ff892",
  ]
}

variable "eks_nodegroup_capacity_type" {
  description = "EKS node group capacity type."
  type        = string
  default     = "ON_DEMAND"
}

variable "eks_nodegroup_ami_type" {
  description = "EKS node group AMI type."
  type        = string
  default     = "AL2023_x86_64_STANDARD"
}

variable "eks_nodegroup_instance_types" {
  description = "EKS node group instance types."
  type        = list(string)
  default = [
    "t3.medium",
  ]
}

variable "eks_nodegroup_disk_size" {
  description = "EKS node group root volume size (GiB)."
  type        = number
  default     = 30
}

variable "eks_nodegroup_min_size" {
  description = "EKS node group min size."
  type        = number
  default     = 0
}

variable "eks_nodegroup_max_size" {
  description = "EKS node group max size."
  type        = number
  default     = 1
}

variable "eks_nodegroup_desired_size" {
  description = "EKS node group desired size."
  type        = number
  default     = 0
}

variable "eks_nodegroup_update_max_unavailable" {
  description = "Maximum unavailable nodes during node group update."
  type        = number
  default     = 1
}

variable "eks_nodegroup_repair_enabled" {
  description = "Whether node group repair is enabled."
  type        = bool
  default     = true
}

variable "enable_eks_ebs_csi_addon" {
  description = "Whether to manage aws-ebs-csi-driver addon."
  type        = bool
  default     = false
}

variable "eks_ebs_csi_addon_version" {
  description = "Optional aws-ebs-csi-driver addon version."
  type        = string
  default     = null
}

variable "enable_eks_cloudwatch_observability_addon" {
  description = "Whether to manage amazon-cloudwatch-observability addon."
  type        = bool
  default     = false
}

variable "eks_cloudwatch_observability_addon_version" {
  description = "Optional amazon-cloudwatch-observability addon version."
  type        = string
  default     = null
}

variable "enable_eks_ebs_csi_pod_identity_association" {
  description = "Whether to manage EBS CSI pod identity association."
  type        = bool
  default     = false
}

variable "eks_ebs_csi_pod_identity_namespace" {
  description = "Namespace for EBS CSI pod identity."
  type        = string
  default     = "kube-system"
}

variable "eks_ebs_csi_pod_identity_service_account" {
  description = "Service account for EBS CSI pod identity."
  type        = string
  default     = "ebs-csi-controller-sa"
}

variable "eks_ebs_csi_role_arn" {
  description = "Role ARN for EBS CSI pod identity. Leave empty to reuse IAM module output."
  type        = string
  default     = "arn:aws:iam::543035741679:role/AmazonEKSPodIdentityAmazonEBSCSIDriverRole"
}

variable "enable_eks_cloudwatch_pod_identity_association" {
  description = "Whether to manage CloudWatch pod identity association."
  type        = bool
  default     = false
}

variable "eks_cloudwatch_pod_identity_namespace" {
  description = "Namespace for CloudWatch pod identity."
  type        = string
  default     = "amazon-cloudwatch"
}

variable "eks_cloudwatch_pod_identity_service_account" {
  description = "Service account for CloudWatch pod identity."
  type        = string
  default     = "cloudwatch-agent"
}

variable "eks_cloudwatch_observability_role_arn" {
  description = "Role ARN for CloudWatch pod identity. Leave empty to reuse IAM module output."
  type        = string
  default     = ""
}
