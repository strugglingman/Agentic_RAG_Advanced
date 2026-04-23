variable "enabled" {
  description = "Whether to manage IAM resources. Keep false until import/apply is intentionally planned."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Common tags."
  type        = map(string)
  default     = {}
}

variable "aws_account_id" {
  description = "AWS account ID used for account-scoped policy ARNs."
  type        = string
}

variable "aws_region" {
  description = "AWS region used for regional policy ARNs."
  type        = string
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
