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

variable "enable_ecr" {
  description = "Whether to manage ECR repositories."
  type        = bool
  default     = false
}

variable "ecr_image_tag_mutability" {
  description = "ECR image tag mutability setting."
  type        = string
  default     = "MUTABLE"
}

variable "ecr_scan_on_push" {
  description = "Whether ECR image scanning on push is enabled."
  type        = bool
  default     = false
}

variable "ecr_encryption_type" {
  description = "ECR encryption type."
  type        = string
  default     = "AES256"
}

variable "ecr_kms_key" {
  description = "KMS key ARN when ECR encryption type is KMS."
  type        = string
  default     = null
  nullable    = true
}

variable "ecr_force_delete" {
  description = "Whether ECR repositories can be force deleted."
  type        = bool
  default     = false
}

variable "ecr_tags" {
  description = "Tags for ECR repositories."
  type        = map(string)
  default     = {}
}

variable "enable_s3_backup_bucket" {
  description = "Whether to manage backup S3 bucket resources."
  type        = bool
  default     = false
}

variable "s3_backup_force_destroy" {
  description = "Whether backup bucket can be force-destroyed."
  type        = bool
  default     = false
}

variable "s3_backup_tags" {
  description = "Tags for backup bucket resources."
  type        = map(string)
  default     = {}
}

variable "s3_backup_manage_public_access_block" {
  description = "Whether to manage backup bucket public access block."
  type        = bool
  default     = true
}

variable "s3_backup_block_public_acls" {
  description = "Backup bucket block public ACLs setting."
  type        = bool
  default     = true
}

variable "s3_backup_ignore_public_acls" {
  description = "Backup bucket ignore public ACLs setting."
  type        = bool
  default     = true
}

variable "s3_backup_block_public_policy" {
  description = "Backup bucket block public policy setting."
  type        = bool
  default     = true
}

variable "s3_backup_restrict_public_buckets" {
  description = "Backup bucket restrict public buckets setting."
  type        = bool
  default     = true
}

variable "s3_backup_manage_server_side_encryption" {
  description = "Whether to manage backup bucket server-side encryption."
  type        = bool
  default     = true
}

variable "s3_backup_bucket_key_enabled" {
  description = "Whether backup bucket key is enabled."
  type        = bool
  default     = true
}

variable "s3_backup_sse_algorithm" {
  description = "Backup bucket SSE algorithm."
  type        = string
  default     = "AES256"
}

variable "s3_backup_kms_master_key_id" {
  description = "KMS key ARN for backup bucket SSE-KMS."
  type        = string
  default     = null
  nullable    = true
}

variable "s3_backup_manage_ownership_controls" {
  description = "Whether to manage backup bucket ownership controls."
  type        = bool
  default     = true
}

variable "s3_backup_object_ownership" {
  description = "Backup bucket object ownership setting."
  type        = string
  default     = "BucketOwnerEnforced"
}

variable "s3_backup_manage_lifecycle_configuration" {
  description = "Whether to manage backup bucket lifecycle configuration."
  type        = bool
  default     = true
}

variable "s3_backup_transition_default_minimum_object_size" {
  description = "Transition minimum object size behavior for lifecycle."
  type        = string
  default     = "all_storage_classes_128K"
}

variable "s3_backup_lifecycle_rule_id" {
  description = "Backup bucket lifecycle rule ID."
  type        = string
  default     = "expire-postgres-backups-after-30-days"
}

variable "s3_backup_lifecycle_prefix" {
  description = "Backup bucket lifecycle prefix."
  type        = string
  default     = "postgres/"
}

variable "s3_backup_lifecycle_expiration_days" {
  description = "Backup bucket lifecycle expiration days."
  type        = number
  default     = 30
}

variable "enable_security_groups" {
  description = "Whether to manage imported EC2/RDS/ElastiCache security groups."
  type        = bool
  default     = false
}

variable "security_groups_vpc_id" {
  description = "VPC ID containing imported security groups."
  type        = string
  default     = "vpc-04fbc127649e7f25e"
}

variable "ec2_security_group_name" {
  description = "EC2 security group name."
  type        = string
  default     = "agentic-rag-web-sg"
}

variable "ec2_security_group_description" {
  description = "EC2 security group description."
  type        = string
  default     = "agentic-rag-web-sg created 2026-04-04T20:33:06.875Z"
}

variable "ec2_security_group_tags" {
  description = "Tags for EC2 security group."
  type        = map(string)
  default     = {}
}

variable "ec2_security_group_ingress_rules" {
  description = "Ingress rules for EC2 security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 80
      to_port          = 80
      protocol         = "tcp"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
    {
      description      = ""
      from_port        = 22
      to_port          = 22
      protocol         = "tcp"
      cidr_blocks      = ["80.217.197.43/32"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
    {
      description      = ""
      from_port        = 443
      to_port          = 443
      protocol         = "tcp"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
  ]
}

variable "ec2_security_group_egress_rules" {
  description = "Egress rules for EC2 security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
  ]
}

variable "rds_security_group_name" {
  description = "RDS security group name."
  type        = string
  default     = "rds-agentic-rag-postgres-sg"
}

variable "rds_security_group_description" {
  description = "RDS security group description."
  type        = string
  default     = "Created by RDS management console"
}

variable "rds_security_group_tags" {
  description = "Tags for RDS security group."
  type        = map(string)
  default     = {}
}

variable "rds_security_group_ingress_rules" {
  description = "Ingress rules for RDS security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 5432
      to_port          = 5432
      protocol         = "tcp"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups = [
        "sg-0a751f46c8106507f",
        "sg-0b1650f56582fdff7",
      ]
      self = false
    },
  ]
}

variable "rds_security_group_egress_rules" {
  description = "Egress rules for RDS security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
  ]
}

variable "elasticache_security_group_name" {
  description = "ElastiCache security group name."
  type        = string
  default     = "agentic-rag-redis-sg"
}

variable "elasticache_security_group_description" {
  description = "ElastiCache security group description."
  type        = string
  default     = "security for connection between application server and redis"
}

variable "elasticache_security_group_tags" {
  description = "Tags for ElastiCache security group."
  type        = map(string)
  default     = {}
}

variable "elasticache_security_group_ingress_rules" {
  description = "Ingress rules for ElastiCache security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 6379
      to_port          = 6379
      protocol         = "tcp"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups = [
        "sg-0b1650f56582fdff7",
        "sg-0a751f46c8106507f",
      ]
      self = false
    },
    {
      description      = ""
      from_port        = 6380
      to_port          = 6380
      protocol         = "tcp"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups = [
        "sg-0b1650f56582fdff7",
        "sg-0a751f46c8106507f",
      ]
      self = false
    },
  ]
}

variable "elasticache_security_group_egress_rules" {
  description = "Egress rules for ElastiCache security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
  ]
}

variable "enable_eks_cluster_security_group" {
  description = "Whether to manage the imported EKS cluster security group."
  type        = bool
  default     = false
}

variable "eks_cluster_security_group_name" {
  description = "EKS cluster security group name."
  type        = string
  default     = "eks-cluster-sg-agentic-rag-eks-788520854"
}

variable "eks_cluster_security_group_description" {
  description = "EKS cluster security group description."
  type        = string
  default     = "EKS created security group applied to ENI that is attached to EKS Control Plane master nodes, as well as any managed workloads."
}

variable "eks_cluster_security_group_tags" {
  description = "Tags for EKS cluster security group."
  type        = map(string)
  default = {
    Name                                    = "eks-cluster-sg-agentic-rag-eks-788520854"
    "kubernetes.io/cluster/agentic-rag-eks" = "owned"
  }
}

variable "eks_cluster_security_group_ingress_rules" {
  description = "Ingress rules for EKS cluster security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = "Allows EFA traffic, which is not matched by CIDR rules."
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = true
    },
    {
      description      = "elbv2.k8s.aws/targetGroupBinding=shared"
      from_port        = 3000
      to_port          = 3000
      protocol         = "tcp"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = ["sg-04e2d6ea7fa2571c6"]
      self             = false
    },
  ]
}

variable "eks_cluster_security_group_egress_rules" {
  description = "Egress rules for EKS cluster security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
    {
      description      = "Allows EFA traffic, which is not matched by CIDR rules."
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = true
    },
  ]
}

variable "enable_default_vpc_security_group" {
  description = "Whether to manage the default VPC security group."
  type        = bool
  default     = false
}

variable "default_vpc_security_group_ingress_rules" {
  description = "Ingress rules for default VPC security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = []
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = true
    },
  ]
}

variable "default_vpc_security_group_egress_rules" {
  description = "Egress rules for default VPC security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
  default = [
    {
      description      = ""
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
  ]
}

variable "default_vpc_security_group_tags" {
  description = "Tags for default VPC security group."
  type        = map(string)
  default     = {}
}

variable "enable_network_baseline" {
  description = "Whether to manage imported default VPC baseline resources (VPC/subnets/IGW/main route table)."
  type        = bool
  default     = false
}

variable "network_vpc_id" {
  description = "Default VPC ID."
  type        = string
  default     = "vpc-04fbc127649e7f25e"
}

variable "network_vpc_enable_dns_support" {
  description = "Whether DNS support is enabled in default VPC."
  type        = bool
  default     = true
}

variable "network_vpc_enable_dns_hostnames" {
  description = "Whether DNS hostnames are enabled in default VPC."
  type        = bool
  default     = true
}

variable "network_vpc_tags" {
  description = "Tags for default VPC."
  type        = map(string)
  default     = {}
}

variable "network_subnet_az_a" {
  description = "AZ name for default subnet A."
  type        = string
  default     = "eu-north-1a"
}

variable "network_subnet_az_a_map_public_ip_on_launch" {
  description = "Whether default subnet A maps public IP on launch."
  type        = bool
  default     = true
}

variable "network_subnet_az_a_tags" {
  description = "Tags for default subnet A."
  type        = map(string)
  default     = {}
}

variable "network_subnet_az_b" {
  description = "AZ name for default subnet B."
  type        = string
  default     = "eu-north-1b"
}

variable "network_subnet_az_b_map_public_ip_on_launch" {
  description = "Whether default subnet B maps public IP on launch."
  type        = bool
  default     = true
}

variable "network_subnet_az_b_tags" {
  description = "Tags for default subnet B."
  type        = map(string)
  default     = {}
}

variable "network_subnet_az_c" {
  description = "AZ name for default subnet C."
  type        = string
  default     = "eu-north-1c"
}

variable "network_subnet_az_c_map_public_ip_on_launch" {
  description = "Whether default subnet C maps public IP on launch."
  type        = bool
  default     = true
}

variable "network_subnet_az_c_tags" {
  description = "Tags for default subnet C."
  type        = map(string)
  default     = {}
}

variable "network_internet_gateway_id" {
  description = "Internet gateway ID for default VPC."
  type        = string
  default     = "igw-0628544cef3ba4d20"
}

variable "network_internet_gateway_tags" {
  description = "Tags for internet gateway."
  type        = map(string)
  default     = {}
}

variable "network_main_route_table_id" {
  description = "Main route table ID in default VPC."
  type        = string
  default     = "rtb-01a4e0c3dbd7b0ff4"
}

variable "network_default_route_cidr_block" {
  description = "Default route CIDR block in main route table."
  type        = string
  default     = "0.0.0.0/0"
}

variable "network_main_route_table_tags" {
  description = "Tags for main route table."
  type        = map(string)
  default     = {}
}

variable "enable_ec2" {
  description = "Whether to manage the legacy EC2 instance."
  type        = bool
  default     = false
}

variable "ec2_ami_id" {
  description = "AMI ID for the legacy EC2 instance."
  type        = string
  default     = "ami-080254318c2d8932f"
}

variable "ec2_instance_type" {
  description = "Instance type for the legacy EC2 instance."
  type        = string
  default     = "t3.small"
}

variable "ec2_subnet_id" {
  description = "Subnet ID used by the legacy EC2 instance."
  type        = string
  default     = "subnet-078fedfabcf3ff892"
}

variable "ec2_vpc_security_group_ids" {
  description = "Security groups attached to the legacy EC2 instance."
  type        = list(string)
  default = [
    "sg-0b1650f56582fdff7",
  ]
}

variable "ec2_key_name" {
  description = "SSH key pair name for the legacy EC2 instance."
  type        = string
  default     = "agentic-rag-eun"
}

variable "ec2_iam_instance_profile_name" {
  description = "IAM instance profile name attached to the legacy EC2 instance."
  type        = string
  default     = "agentic-rag-ec2-s3-backup-role"
}

variable "ec2_monitoring" {
  description = "Whether detailed monitoring is enabled on EC2."
  type        = bool
  default     = false
}

variable "ec2_ebs_optimized" {
  description = "Whether EBS optimization is enabled on EC2."
  type        = bool
  default     = true
}

variable "ec2_metadata_http_endpoint" {
  description = "IMDS endpoint mode for EC2."
  type        = string
  default     = "enabled"
}

variable "ec2_metadata_http_tokens" {
  description = "IMDS token requirement for EC2."
  type        = string
  default     = "required"
}

variable "ec2_metadata_http_put_response_hop_limit" {
  description = "IMDS hop limit for EC2."
  type        = number
  default     = 2
}

variable "ec2_root_block_device_delete_on_termination" {
  description = "Whether the EC2 root volume is deleted on termination."
  type        = bool
  default     = true
}

variable "ec2_root_block_device_volume_size" {
  description = "EC2 root volume size (GiB)."
  type        = number
  default     = 30
}

variable "ec2_root_block_device_volume_type" {
  description = "EC2 root volume type."
  type        = string
  default     = "gp3"
}

variable "ec2_root_block_device_iops" {
  description = "EC2 root volume IOPS."
  type        = number
  default     = 3000
}

variable "ec2_root_block_device_throughput" {
  description = "EC2 root volume throughput (MiB/s)."
  type        = number
  default     = 125
}

variable "ec2_tags" {
  description = "Tags for the legacy EC2 instance. Keep legacy tags for no-op adoption."
  type        = map(string)
  default = {
    Name = "agentic-rag"
  }
}

variable "enable_rds" {
  description = "Whether to manage the existing RDS instance."
  type        = bool
  default     = false
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
  description = "RDS master password. Keep null for import-first unless password rotation is planned."
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
  description = "RDS allocated storage in GiB."
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS max allocated storage autoscaling limit in GiB."
  type        = number
  default     = 1000
}

variable "rds_storage_type" {
  description = "RDS storage type."
  type        = string
  default     = "gp3"
}

variable "rds_iops" {
  description = "RDS storage IOPS."
  type        = number
  default     = 3000
}

variable "rds_storage_throughput" {
  description = "RDS storage throughput in MiB/s."
  type        = number
  default     = 125
}

variable "rds_storage_encrypted" {
  description = "Whether RDS storage encryption is enabled."
  type        = bool
  default     = true
}

variable "rds_kms_key_id" {
  description = "KMS key ARN for RDS storage encryption."
  type        = string
  default     = "arn:aws:kms:eu-north-1:543035741679:key/458f3d51-30d6-4052-9475-91f9923e4427"
}

variable "rds_db_subnet_group_name" {
  description = "RDS DB subnet group name."
  type        = string
  default     = "default-vpc-04fbc127649e7f25e"
}

variable "rds_vpc_security_group_ids" {
  description = "Security groups attached to the RDS instance."
  type        = list(string)
  default = [
    "sg-022a4409eb6f0ea18",
  ]
}

variable "rds_parameter_group_name" {
  description = "RDS parameter group name."
  type        = string
  default     = "default.postgres15"
}

variable "rds_publicly_accessible" {
  description = "Whether RDS is publicly accessible."
  type        = bool
  default     = false
}

variable "rds_multi_az" {
  description = "Whether RDS is deployed Multi-AZ."
  type        = bool
  default     = false
}

variable "rds_auto_minor_version_upgrade" {
  description = "Whether RDS auto minor version upgrades are enabled."
  type        = bool
  default     = true
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period in days."
  type        = number
  default     = 1
}

variable "rds_backup_window" {
  description = "RDS preferred backup window."
  type        = string
  default     = "03:38-04:08"
}

variable "rds_maintenance_window" {
  description = "RDS preferred maintenance window."
  type        = string
  default     = "sun:23:42-mon:00:12"
}

variable "rds_copy_tags_to_snapshot" {
  description = "Whether RDS copies tags to snapshots."
  type        = bool
  default     = true
}

variable "rds_deletion_protection" {
  description = "Whether RDS deletion protection is enabled."
  type        = bool
  default     = false
}

variable "rds_monitoring_interval" {
  description = "RDS enhanced monitoring interval."
  type        = number
  default     = 0
}

variable "rds_performance_insights_enabled" {
  description = "Whether RDS Performance Insights is enabled."
  type        = bool
  default     = false
}

variable "rds_iam_database_authentication_enabled" {
  description = "Whether RDS IAM database authentication is enabled."
  type        = bool
  default     = false
}

variable "rds_ca_cert_identifier" {
  description = "RDS CA certificate identifier."
  type        = string
  default     = "rds-ca-rsa2048-g1"
}

variable "rds_network_type" {
  description = "RDS network type."
  type        = string
  default     = "IPV4"
}

variable "rds_apply_immediately" {
  description = "Whether RDS modifications should be applied immediately."
  type        = bool
  default     = false
}

variable "rds_skip_final_snapshot" {
  description = "Whether to skip final snapshot during DB deletion. Keep aligned with imported baseline."
  type        = bool
  default     = true
}

variable "rds_tags" {
  description = "Tags for the RDS instance."
  type        = map(string)
  default     = {}
}

variable "enable_elasticache" {
  description = "Whether to manage the existing ElastiCache serverless cache."
  type        = bool
  default     = false
}

variable "elasticache_serverless_cache_name" {
  description = "ElastiCache serverless cache name."
  type        = string
  default     = "agentic-rag-redis"
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
  description = "ElastiCache serverless description."
  type        = string
  default     = " "
}

variable "elasticache_kms_key_id" {
  description = "Optional KMS key ARN for ElastiCache serverless."
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

variable "elasticache_security_group_ids" {
  description = "Security groups attached to ElastiCache serverless."
  type        = list(string)
  default = [
    "sg-019b31f41c90538b6",
  ]
}

variable "elasticache_subnet_ids" {
  description = "Subnets attached to ElastiCache serverless."
  type        = list(string)
  default = [
    "subnet-078fedfabcf3ff892",
    "subnet-05159408d4ed93c97",
  ]
}

variable "elasticache_snapshot_retention_limit" {
  description = "ElastiCache snapshot retention limit in days."
  type        = number
  default     = 0
}

variable "elasticache_daily_snapshot_time" {
  description = "ElastiCache daily snapshot time in UTC."
  type        = string
  default     = "03:30"
}

variable "elasticache_tags" {
  description = "Tags for ElastiCache serverless."
  type        = map(string)
  default     = {}
}

variable "enable_acm" {
  description = "Whether to manage the existing ACM certificate."
  type        = bool
  default     = false
}

variable "acm_certificate_domain_name" {
  description = "Primary domain name for ACM certificate."
  type        = string
  default     = "wxwlabs.pro"
}

variable "acm_certificate_subject_alternative_names" {
  description = "Subject alternative names for ACM certificate."
  type        = list(string)
  default = [
    "www.wxwlabs.pro",
  ]
}

variable "acm_validation_method" {
  description = "Validation method for ACM certificate."
  type        = string
  default     = "DNS"
}

variable "acm_key_algorithm" {
  description = "Public key algorithm for ACM certificate."
  type        = string
  default     = "RSA_2048"
}

variable "acm_certificate_transparency_logging_preference" {
  description = "ACM certificate transparency logging preference."
  type        = string
  default     = "ENABLED"
}

variable "acm_tags" {
  description = "Tags for ACM certificate."
  type        = map(string)
  default     = {}
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
