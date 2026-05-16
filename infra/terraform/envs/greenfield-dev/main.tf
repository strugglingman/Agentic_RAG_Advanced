data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_kms_alias" "rds_input" {
  count = startswith(var.rds_kms_key_id, "alias/") ? 1 : 0
  name  = var.rds_kms_key_id
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.extra_tags
  )

  selected_azs = slice(data.aws_availability_zones.available.names, 0, var.az_count)

  public_subnet_map = {
    for idx, az in local.selected_azs : az => {
      az   = az
      cidr = cidrsubnet(var.vpc_cidr, var.public_subnet_newbits, idx)
    }
  }

  private_subnet_map = {
    for idx, az in local.selected_azs : az => {
      az   = az
      cidr = cidrsubnet(var.vpc_cidr, var.private_subnet_newbits, idx + 8)
    }
  }

  public_subnet_ids  = [for az in sort(keys(aws_subnet.public)) : aws_subnet.public[az].id]
  private_subnet_ids = [for az in sort(keys(aws_subnet.private)) : aws_subnet.private[az].id]

  s3_backup_bucket_name  = var.s3_backup_bucket_name != "" ? var.s3_backup_bucket_name : "${var.project_name}-${var.environment}-backup-${data.aws_caller_identity.current.account_id}-${var.aws_region}"
  ecr_backend_repo_name  = var.ecr_backend_repo_name != "" ? var.ecr_backend_repo_name : "${var.project_name}-backend"
  ecr_frontend_repo_name = var.ecr_frontend_repo_name != "" ? var.ecr_frontend_repo_name : "${var.project_name}-frontend"
  eks_cluster_name       = var.eks_cluster_name != "" ? var.eks_cluster_name : "${local.name_prefix}-eks"
  rds_identifier         = var.rds_instance_identifier != "" ? var.rds_instance_identifier : "${local.name_prefix}-postgres"
  elasticache_name       = var.elasticache_serverless_cache_name != "" ? var.elasticache_serverless_cache_name : "${local.name_prefix}-redis"
  sns_topic_name         = var.sns_topic_name != "" ? var.sns_topic_name : "${local.name_prefix}-alerts"

  eks_cluster_role_name   = "${local.name_prefix}-eks-cluster-role"
  eks_node_role_name      = "${local.name_prefix}-eks-node-role"
  eks_ebs_csi_role_name   = "${local.name_prefix}-eks-ebs-csi-role"
  eks_cw_role_name        = "${local.name_prefix}-eks-cloudwatch-observability-role"
  ec2_backup_role_name    = "${local.name_prefix}-ec2-backup-role"
  ec2_instance_profile    = "${local.name_prefix}-ec2-backup-profile"
  s3_backup_policy_name   = "${local.name_prefix}-s3-backup-policy"
  ecr_push_policy_name    = "${local.name_prefix}-ecr-push-policy"
  ec2_security_group_name = "${local.name_prefix}-legacy-ec2-sg"

  eks_cluster_role_arn_effective = module.iam.eks_cluster_role_arn
  eks_node_role_arn_effective    = module.iam.eks_node_role_arn
  eks_ebs_csi_role_arn_effective = module.iam.eks_ebs_csi_role_arn
  eks_cw_role_arn_effective      = module.iam.eks_cloudwatch_observability_role_arn

  monitoring_rds_instance_id = var.rds_instance_id_for_monitoring != "" ? var.rds_instance_id_for_monitoring : local.rds_identifier
  redis_alarm_dimensions = length(var.redis_alarm_dimensions_override) > 0 ? var.redis_alarm_dimensions_override : {
    CacheClusterId = local.elasticache_name
  }
  redis_engine_alarm_dimensions = length(var.redis_engine_alarm_dimensions_override) > 0 ? var.redis_engine_alarm_dimensions_override : {
    CacheName = local.elasticache_name
  }

  rds_kms_key_id_effective = startswith(var.rds_kms_key_id, "alias/") ? data.aws_kms_alias.rds_input[0].target_key_arn : var.rds_kms_key_id

  eks_admin_entries = {
    for idx, arn in var.eks_admin_principal_arns : "admin_${idx}" => {
      principal_arn     = arn
      type              = "STANDARD"
      kubernetes_groups = []
      username          = null
      tags              = {}
      policy_associations = [
        {
          policy_arn        = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope_type = "cluster"
          namespaces        = []
        }
      ]
    }
  }

  eks_access_entries_effective = merge(local.eks_admin_entries, var.additional_eks_access_entries)

  ec2_security_group_ingress_rules = var.enable_legacy_ec2 ? [
    {
      description      = "SSH access"
      from_port        = 22
      to_port          = 22
      protocol         = "tcp"
      cidr_blocks      = var.ec2_ingress_cidrs
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
    {
      description      = "HTTP access"
      from_port        = 80
      to_port          = 80
      protocol         = "tcp"
      cidr_blocks      = var.ec2_ingress_cidrs
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
    {
      description      = "HTTPS access"
      from_port        = 443
      to_port          = 443
      protocol         = "tcp"
      cidr_blocks      = var.ec2_ingress_cidrs
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    }
  ] : []

  default_allow_all_egress = [
    {
      description      = "Allow all outbound traffic"
      from_port        = 0
      to_port          = 0
      protocol         = "-1"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    }
  ]

  rds_security_group_ingress_rules = [
    {
      description      = "PostgreSQL from VPC CIDR"
      from_port        = 5432
      to_port          = 5432
      protocol         = "tcp"
      cidr_blocks      = [var.vpc_cidr]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    }
  ]

  elasticache_security_group_ingress_rules = [
    {
      description      = "Valkey/Redis from VPC CIDR"
      from_port        = 6379
      to_port          = 6379
      protocol         = "tcp"
      cidr_blocks      = [var.vpc_cidr]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    },
    {
      description      = "TLS Valkey/Redis from VPC CIDR"
      from_port        = 6380
      to_port          = 6380
      protocol         = "tcp"
      cidr_blocks      = [var.vpc_cidr]
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      security_groups  = []
      self             = false
    }
  ]

  security_groups_enabled = var.enable_security_groups && (var.enable_rds || var.enable_elasticache || var.enable_legacy_ec2)

  rds_security_group_id_effective         = local.security_groups_enabled ? module.security_groups.rds_security_group_id : null
  elasticache_security_group_id_effective = local.security_groups_enabled ? module.security_groups.elasticache_security_group_id : null
  legacy_ec2_security_group_id_effective  = local.security_groups_enabled ? module.security_groups.ec2_security_group_id : null
}

check "az_count_supported" {
  assert {
    condition     = length(data.aws_availability_zones.available.names) >= var.az_count
    error_message = "Current region does not have enough AZs for az_count."
  }
}

check "eks_endpoint_mode_valid" {
  assert {
    condition     = !var.enable_eks || var.enable_eks_public_endpoint || var.enable_eks_private_endpoint
    error_message = "At least one EKS endpoint access mode must be enabled."
  }
}

check "eks_public_cidrs_required" {
  assert {
    condition     = !var.enable_eks || !var.enable_eks_public_endpoint || length(var.eks_public_access_cidrs) > 0
    error_message = "Set eks_public_access_cidrs when enable_eks_public_endpoint=true."
  }
}

check "rds_password_required" {
  assert {
    condition     = !var.enable_rds || (var.rds_master_password != null && trimspace(var.rds_master_password) != "")
    error_message = "Set rds_master_password in terraform.tfvars.local when enable_rds=true."
  }
}

check "acm_domain_required" {
  assert {
    condition     = !var.enable_acm || trimspace(var.acm_domain_name) != ""
    error_message = "Set acm_domain_name when enable_acm=true."
  }
}

check "eks_role_source_required" {
  assert {
    condition     = !var.enable_eks || var.enable_iam
    error_message = "enable_iam must be true when enable_eks=true in greenfield mode."
  }
}

check "eks_ebs_role_source_required" {
  assert {
    condition     = !var.enable_eks || !var.enable_eks_ebs_csi_pod_identity_association || var.enable_iam
    error_message = "enable_iam must be true when EBS CSI pod identity association is enabled."
  }
}

check "legacy_ec2_inputs" {
  assert {
    condition = !var.enable_legacy_ec2 || (
      trimspace(var.ec2_ami_id) != "" &&
      trimspace(var.ec2_key_name) != "" &&
      length(var.ec2_ingress_cidrs) > 0 &&
      var.enable_iam
    )
    error_message = "Legacy EC2 requires ec2_ami_id, ec2_key_name, ec2_ingress_cidrs, and enable_iam=true."
  }
}

check "security_groups_required_for_runtime" {
  assert {
    condition     = var.enable_security_groups || (!var.enable_rds && !var.enable_elasticache && !var.enable_legacy_ec2)
    error_message = "enable_security_groups must be true when RDS/ElastiCache/legacy EC2 are enabled in greenfield root."
  }
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc"
  })
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-igw"
  })
}

resource "aws_subnet" "public" {
  for_each = local.public_subnet_map

  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.value.az
  cidr_block              = each.value.cidr
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name                     = "${local.name_prefix}-public-${each.value.az}"
    Tier                     = "public"
    "kubernetes.io/role/elb" = "1"
  })
}

resource "aws_subnet" "private" {
  for_each = local.private_subnet_map

  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.value.az
  cidr_block              = each.value.cidr
  map_public_ip_on_launch = false

  tags = merge(local.common_tags, {
    Name                              = "${local.name_prefix}-private-${each.value.az}"
    Tier                              = "private"
    "kubernetes.io/role/internal-elb" = "1"
  })
}

resource "aws_eip" "nat" {
  count = var.enable_nat_gateway ? 1 : 0

  domain = "vpc"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-nat-eip"
  })
}

resource "aws_nat_gateway" "this" {
  count = var.enable_nat_gateway ? 1 : 0

  allocation_id = aws_eip.nat[0].id
  subnet_id     = local.public_subnet_ids[0]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-nat"
  })

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-rt"
  })
}

resource "aws_route" "public_default" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-private-rt"
  })
}

resource "aws_route" "private_default" {
  count = var.enable_nat_gateway ? 1 : 0

  route_table_id         = aws_route_table.private.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.this[0].id
}

resource "aws_route_table_association" "private" {
  for_each = aws_subnet.private

  subnet_id      = each.value.id
  route_table_id = aws_route_table.private.id
}

resource "aws_db_subnet_group" "this" {
  count = var.enable_rds ? 1 : 0

  name       = "${local.name_prefix}-rds-subnets"
  subnet_ids = local.private_subnet_ids

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-subnets"
  })
}

module "security_groups" {
  source = "../../modules/security_groups"

  enabled = local.security_groups_enabled
  vpc_id  = aws_vpc.this.id

  ec2_name          = local.ec2_security_group_name
  ec2_description   = "Legacy EC2 host access"
  ec2_tags          = local.common_tags
  ec2_ingress_rules = local.ec2_security_group_ingress_rules
  ec2_egress_rules  = local.default_allow_all_egress

  rds_name          = "${local.name_prefix}-rds-sg"
  rds_description   = "RDS access from VPC"
  rds_tags          = local.common_tags
  rds_ingress_rules = local.rds_security_group_ingress_rules
  rds_egress_rules  = local.default_allow_all_egress

  elasticache_name          = "${local.name_prefix}-elasticache-sg"
  elasticache_description   = "ElastiCache access from VPC"
  elasticache_tags          = local.common_tags
  elasticache_ingress_rules = local.elasticache_security_group_ingress_rules
  elasticache_egress_rules  = local.default_allow_all_egress

  enable_eks_cluster_security_group = false
  enable_default_vpc_security_group = false
}

module "iam" {
  source = "../../modules/iam"

  enabled = var.enable_iam
  tags    = local.common_tags

  aws_account_id = data.aws_caller_identity.current.account_id
  aws_region     = var.aws_region

  eks_cluster_role_name                    = local.eks_cluster_role_name
  eks_node_role_name                       = local.eks_node_role_name
  eks_ebs_csi_role_name                    = local.eks_ebs_csi_role_name
  eks_cloudwatch_observability_role_name   = local.eks_cw_role_name
  enable_eks_ebs_csi_role                  = var.enable_eks_ebs_csi_role
  enable_eks_cloudwatch_observability_role = var.enable_eks_cloudwatch_observability_role

  ec2_backup_role_name         = local.ec2_backup_role_name
  ec2_backup_role_description  = "Legacy EC2 role for backup and image push operations."
  ec2_instance_profile_name    = local.ec2_instance_profile
  s3_backup_policy_name        = local.s3_backup_policy_name
  s3_backup_policy_description = "Allow access to backup bucket objects."
  ecr_push_policy_name         = local.ecr_push_policy_name
  ecr_push_policy_description  = "Allow push/pull on project ECR repos."

  s3_backup_bucket_name  = local.s3_backup_bucket_name
  ecr_backend_repo_name  = local.ecr_backend_repo_name
  ecr_frontend_repo_name = local.ecr_frontend_repo_name
}

module "ecr" {
  source = "../../modules/ecr"

  enabled = var.enable_ecr

  backend_repository_name  = local.ecr_backend_repo_name
  frontend_repository_name = local.ecr_frontend_repo_name
  image_tag_mutability     = var.ecr_image_tag_mutability
  scan_on_push             = var.ecr_scan_on_push
  encryption_type          = var.ecr_encryption_type
  kms_key                  = var.ecr_kms_key
  force_delete             = var.ecr_force_delete
  tags                     = local.common_tags
}

module "s3_backup" {
  source = "../../modules/s3_backup"

  enabled = var.enable_s3_backup_bucket

  bucket_name   = local.s3_backup_bucket_name
  force_destroy = var.s3_backup_force_destroy
  tags          = local.common_tags

  manage_public_access_block = true
  block_public_acls          = true
  ignore_public_acls         = true
  block_public_policy        = true
  restrict_public_buckets    = true

  manage_server_side_encryption = true
  bucket_key_enabled            = true
  sse_algorithm                 = var.s3_backup_sse_algorithm
  kms_master_key_id             = var.s3_backup_kms_master_key_id

  manage_ownership_controls = true
  object_ownership          = "BucketOwnerEnforced"

  manage_lifecycle_configuration         = true
  transition_default_minimum_object_size = "all_storage_classes_128K"
  lifecycle_rule_id                      = "expire-postgres-backups"
  lifecycle_prefix                       = "postgres/"
  lifecycle_expiration_days              = var.s3_backup_lifecycle_expiration_days
}

module "eks" {
  source = "../../modules/eks"

  enabled      = var.enable_eks
  tags         = local.common_tags
  cluster_tags = local.common_tags

  cluster_name                          = local.eks_cluster_name
  cluster_role_arn                      = local.eks_cluster_role_arn_effective
  cluster_version                       = var.eks_cluster_version
  cluster_bootstrap_self_managed_addons = var.eks_cluster_bootstrap_self_managed_addons
  cluster_subnet_ids                    = local.private_subnet_ids
  cluster_endpoint_public_access        = var.enable_eks_public_endpoint
  cluster_endpoint_private_access       = var.enable_eks_private_endpoint
  cluster_public_access_cidrs           = var.eks_public_access_cidrs
  cluster_authentication_mode           = var.eks_cluster_authentication_mode
  cluster_bootstrap_admin_permissions   = var.eks_cluster_bootstrap_admin_permissions
  cluster_ip_family                     = "ipv4"
  cluster_upgrade_support_type          = "STANDARD"
  cluster_zonal_shift_enabled           = false
  cluster_enabled_log_types             = var.eks_cluster_enabled_log_types

  nodegroup_enabled                = var.enable_eks_nodegroup
  nodegroup_name                   = "${local.name_prefix}-${var.eks_nodegroup_name}"
  nodegroup_role_arn               = local.eks_node_role_arn_effective
  nodegroup_subnet_ids             = local.private_subnet_ids
  nodegroup_capacity_type          = var.eks_nodegroup_capacity_type
  nodegroup_ami_type               = var.eks_nodegroup_ami_type
  nodegroup_instance_types         = var.eks_nodegroup_instance_types
  nodegroup_disk_size              = var.eks_nodegroup_disk_size
  nodegroup_min_size               = var.eks_nodegroup_min_size
  nodegroup_max_size               = var.eks_nodegroup_max_size
  nodegroup_desired_size           = var.eks_nodegroup_desired_size
  nodegroup_update_max_unavailable = var.eks_nodegroup_update_max_unavailable
  nodegroup_repair_enabled         = true

  enable_ebs_csi_addon                       = var.enable_eks_ebs_csi_addon
  ebs_csi_addon_version                      = var.eks_ebs_csi_addon_version
  enable_cloudwatch_observability_addon      = var.enable_eks_cloudwatch_observability_addon
  cloudwatch_observability_addon_version     = var.eks_cloudwatch_observability_addon_version
  enable_ebs_csi_pod_identity_association    = var.enable_eks_ebs_csi_pod_identity_association
  ebs_csi_pod_identity_namespace             = "kube-system"
  ebs_csi_pod_identity_service_account       = "ebs-csi-controller-sa"
  ebs_csi_pod_identity_role_arn              = local.eks_ebs_csi_role_arn_effective
  enable_cloudwatch_pod_identity_association = var.enable_eks_cloudwatch_pod_identity_association
  cloudwatch_pod_identity_namespace          = "amazon-cloudwatch"
  cloudwatch_pod_identity_service_account    = "cloudwatch-agent"
  cloudwatch_pod_identity_role_arn           = local.eks_cw_role_arn_effective
}

module "eks_access" {
  source = "../../modules/eks_access"

  enabled        = var.enable_eks_access_entries && var.enable_eks && length(local.eks_access_entries_effective) > 0
  cluster_name   = local.eks_cluster_name
  access_entries = local.eks_access_entries_effective

  depends_on = [module.eks]
}

module "rds" {
  source = "../../modules/rds"

  enabled                = var.enable_rds
  db_instance_identifier = local.rds_identifier

  engine         = var.rds_engine
  engine_version = var.rds_engine_version
  instance_class = var.rds_instance_class

  db_name  = var.rds_db_name
  username = var.rds_master_username
  password = var.rds_master_password
  port     = var.rds_port

  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type          = var.rds_storage_type
  iops                  = var.rds_iops
  storage_throughput    = var.rds_storage_throughput
  storage_encrypted     = true
  kms_key_id            = local.rds_kms_key_id_effective

  db_subnet_group_name   = aws_db_subnet_group.this[0].name
  vpc_security_group_ids = local.rds_security_group_id_effective != null ? [local.rds_security_group_id_effective] : []
  parameter_group_name   = var.rds_parameter_group_name

  publicly_accessible                 = false
  multi_az                            = var.rds_multi_az
  auto_minor_version_upgrade          = true
  backup_retention_period             = var.rds_backup_retention_period
  backup_window                       = var.rds_backup_window
  maintenance_window                  = var.rds_maintenance_window
  copy_tags_to_snapshot               = true
  deletion_protection                 = var.rds_deletion_protection
  monitoring_interval                 = var.rds_monitoring_interval
  performance_insights_enabled        = var.rds_performance_insights_enabled
  iam_database_authentication_enabled = var.rds_iam_database_authentication_enabled
  ca_cert_identifier                  = var.rds_ca_cert_identifier
  network_type                        = "IPV4"
  apply_immediately                   = var.rds_apply_immediately
  skip_final_snapshot                 = var.rds_skip_final_snapshot

  tags = local.common_tags
}

module "elasticache" {
  source = "../../modules/elasticache"

  enabled = var.enable_elasticache

  serverless_cache_name = local.elasticache_name
  engine                = var.elasticache_engine
  major_engine_version  = var.elasticache_major_engine_version
  description           = var.elasticache_description
  kms_key_id            = var.elasticache_kms_key_id
  user_group_id         = var.elasticache_user_group_id

  security_group_ids       = local.elasticache_security_group_id_effective != null ? [local.elasticache_security_group_id_effective] : []
  subnet_ids               = local.private_subnet_ids
  snapshot_retention_limit = var.elasticache_snapshot_retention_limit
  daily_snapshot_time      = var.elasticache_daily_snapshot_time

  tags = local.common_tags
}

module "acm" {
  source = "../../modules/acm"

  enabled = var.enable_acm

  domain_name               = var.acm_domain_name
  subject_alternative_names = var.acm_subject_alternative_names
  validation_method         = var.acm_validation_method
  key_algorithm             = var.acm_key_algorithm

  certificate_transparency_logging_preference = var.acm_certificate_transparency_logging_preference
  tags                                        = local.common_tags
}

module "monitoring" {
  source = "../../modules/monitoring"

  enabled = var.enable_monitoring
  tags    = local.common_tags

  sns_topic_name                = local.sns_topic_name
  alert_email_endpoint          = var.alert_email_endpoint
  rds_instance_id               = local.monitoring_rds_instance_id
  redis_alarm_dimensions        = local.redis_alarm_dimensions
  redis_engine_alarm_dimensions = local.redis_engine_alarm_dimensions
  alarm_name_prefix             = local.name_prefix
  eks_namespace                 = var.eks_namespace
}

module "ec2" {
  source = "../../modules/ec2"

  enabled = var.enable_legacy_ec2

  ami_id                    = var.ec2_ami_id
  instance_type             = var.ec2_instance_type
  subnet_id                 = local.public_subnet_ids[0]
  vpc_security_group_ids    = local.legacy_ec2_security_group_id_effective != null ? [local.legacy_ec2_security_group_id_effective] : []
  key_name                  = var.ec2_key_name
  iam_instance_profile_name = local.ec2_instance_profile
  monitoring                = false
  ebs_optimized             = true

  metadata_http_endpoint               = "enabled"
  metadata_http_tokens                 = "required"
  metadata_http_put_response_hop_limit = 2

  root_block_device_delete_on_termination = true
  root_block_device_volume_size           = 30
  root_block_device_volume_type           = "gp3"
  root_block_device_iops                  = 3000
  root_block_device_throughput            = 125

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-legacy-ec2"
  })
}
