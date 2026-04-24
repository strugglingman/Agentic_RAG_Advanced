locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  eks_cluster_role_arn_effective           = var.eks_cluster_role_arn != "" ? var.eks_cluster_role_arn : try(module.iam.eks_cluster_role_arn, "")
  eks_node_role_arn_effective              = var.eks_node_role_arn != "" ? var.eks_node_role_arn : try(module.iam.eks_node_role_arn, "")
  eks_ebs_csi_role_arn_effective           = var.eks_ebs_csi_role_arn != "" ? var.eks_ebs_csi_role_arn : try(module.iam.eks_ebs_csi_role_arn, "")
  eks_cw_role_arn_effective                = var.eks_cloudwatch_observability_role_arn != "" ? var.eks_cloudwatch_observability_role_arn : try(module.iam.eks_cloudwatch_observability_role_arn, "")
  ec2_security_group_ids_effective         = var.enable_security_groups && try(module.security_groups.ec2_security_group_id, null) != null ? [module.security_groups.ec2_security_group_id] : var.ec2_vpc_security_group_ids
  rds_security_group_ids_effective         = var.enable_security_groups && try(module.security_groups.rds_security_group_id, null) != null ? [module.security_groups.rds_security_group_id] : var.rds_vpc_security_group_ids
  elasticache_security_group_ids_effective = var.enable_security_groups && try(module.security_groups.elasticache_security_group_id, null) != null ? [module.security_groups.elasticache_security_group_id] : var.elasticache_security_group_ids
}

module "monitoring" {
  source = "../../modules/monitoring"

  enabled                       = var.enable_monitoring
  tags                          = local.common_tags
  sns_topic_name                = var.sns_topic_name
  alarm_name_prefix             = var.project_name
  alert_email_endpoint          = var.alert_email_endpoint
  eks_namespace                 = var.eks_namespace
  rds_instance_id               = var.rds_instance_identifier
  redis_alarm_dimensions        = var.redis_alarm_dimensions
  redis_engine_alarm_dimensions = var.redis_engine_alarm_dimensions
}

module "iam" {
  source = "../../modules/iam"

  enabled = var.enable_iam
  tags    = var.iam_tags

  aws_account_id = var.aws_account_id
  aws_region     = var.aws_region

  eks_cluster_role_name                    = var.eks_cluster_role_name
  eks_node_role_name                       = var.eks_node_role_name
  eks_ebs_csi_role_name                    = var.eks_ebs_csi_role_name
  eks_cloudwatch_observability_role_name   = var.eks_cloudwatch_observability_role_name
  enable_eks_ebs_csi_role                  = var.enable_eks_ebs_csi_role
  enable_eks_cloudwatch_observability_role = var.enable_eks_cloudwatch_observability_role
  ec2_backup_role_name                     = var.ec2_backup_role_name
  ec2_backup_role_description              = var.ec2_backup_role_description
  ec2_instance_profile_name                = var.ec2_instance_profile_name
  s3_backup_policy_name                    = var.s3_backup_policy_name
  s3_backup_policy_description             = var.s3_backup_policy_description
  ecr_push_policy_name                     = var.ecr_push_policy_name
  ecr_push_policy_description              = var.ecr_push_policy_description
  s3_backup_bucket_name                    = var.s3_backup_bucket_name
  ecr_backend_repo_name                    = var.ecr_backend_repo_name
  ecr_frontend_repo_name                   = var.ecr_frontend_repo_name
}

module "ecr" {
  source = "../../modules/ecr"

  enabled = var.enable_ecr

  backend_repository_name  = var.ecr_backend_repo_name
  frontend_repository_name = var.ecr_frontend_repo_name
  image_tag_mutability     = var.ecr_image_tag_mutability
  scan_on_push             = var.ecr_scan_on_push
  encryption_type          = var.ecr_encryption_type
  kms_key                  = var.ecr_kms_key
  force_delete             = var.ecr_force_delete
  tags                     = var.ecr_tags
}

module "s3_backup" {
  source = "../../modules/s3_backup"

  enabled = var.enable_s3_backup_bucket

  bucket_name   = var.s3_backup_bucket_name
  force_destroy = var.s3_backup_force_destroy
  tags          = var.s3_backup_tags

  manage_public_access_block = var.s3_backup_manage_public_access_block
  block_public_acls          = var.s3_backup_block_public_acls
  ignore_public_acls         = var.s3_backup_ignore_public_acls
  block_public_policy        = var.s3_backup_block_public_policy
  restrict_public_buckets    = var.s3_backup_restrict_public_buckets

  manage_server_side_encryption = var.s3_backup_manage_server_side_encryption
  bucket_key_enabled            = var.s3_backup_bucket_key_enabled
  sse_algorithm                 = var.s3_backup_sse_algorithm
  kms_master_key_id             = var.s3_backup_kms_master_key_id

  manage_ownership_controls = var.s3_backup_manage_ownership_controls
  object_ownership          = var.s3_backup_object_ownership

  manage_lifecycle_configuration         = var.s3_backup_manage_lifecycle_configuration
  transition_default_minimum_object_size = var.s3_backup_transition_default_minimum_object_size
  lifecycle_rule_id                      = var.s3_backup_lifecycle_rule_id
  lifecycle_prefix                       = var.s3_backup_lifecycle_prefix
  lifecycle_expiration_days              = var.s3_backup_lifecycle_expiration_days
}

module "security_groups" {
  source = "../../modules/security_groups"

  enabled = var.enable_security_groups

  vpc_id = var.security_groups_vpc_id

  ec2_name          = var.ec2_security_group_name
  ec2_description   = var.ec2_security_group_description
  ec2_tags          = var.ec2_security_group_tags
  ec2_ingress_rules = var.ec2_security_group_ingress_rules
  ec2_egress_rules  = var.ec2_security_group_egress_rules

  rds_name          = var.rds_security_group_name
  rds_description   = var.rds_security_group_description
  rds_tags          = var.rds_security_group_tags
  rds_ingress_rules = var.rds_security_group_ingress_rules
  rds_egress_rules  = var.rds_security_group_egress_rules

  elasticache_name          = var.elasticache_security_group_name
  elasticache_description   = var.elasticache_security_group_description
  elasticache_tags          = var.elasticache_security_group_tags
  elasticache_ingress_rules = var.elasticache_security_group_ingress_rules
  elasticache_egress_rules  = var.elasticache_security_group_egress_rules

  enable_eks_cluster_security_group = var.enable_eks_cluster_security_group
  eks_cluster_name                  = var.eks_cluster_security_group_name
  eks_cluster_description           = var.eks_cluster_security_group_description
  eks_cluster_tags                  = var.eks_cluster_security_group_tags
  eks_cluster_ingress_rules         = var.eks_cluster_security_group_ingress_rules
  eks_cluster_egress_rules          = var.eks_cluster_security_group_egress_rules

  enable_default_vpc_security_group = var.enable_default_vpc_security_group
  default_vpc_ingress_rules         = var.default_vpc_security_group_ingress_rules
  default_vpc_egress_rules          = var.default_vpc_security_group_egress_rules
  default_vpc_tags                  = var.default_vpc_security_group_tags
}

module "network_baseline" {
  source = "../../modules/network_baseline"

  enabled = var.enable_network_baseline

  vpc_id                   = var.network_vpc_id
  vpc_enable_dns_support   = var.network_vpc_enable_dns_support
  vpc_enable_dns_hostnames = var.network_vpc_enable_dns_hostnames
  vpc_tags                 = var.network_vpc_tags

  subnet_az_a                         = var.network_subnet_az_a
  subnet_az_a_map_public_ip_on_launch = var.network_subnet_az_a_map_public_ip_on_launch
  subnet_az_a_tags                    = var.network_subnet_az_a_tags

  subnet_az_b                         = var.network_subnet_az_b
  subnet_az_b_map_public_ip_on_launch = var.network_subnet_az_b_map_public_ip_on_launch
  subnet_az_b_tags                    = var.network_subnet_az_b_tags

  subnet_az_c                         = var.network_subnet_az_c
  subnet_az_c_map_public_ip_on_launch = var.network_subnet_az_c_map_public_ip_on_launch
  subnet_az_c_tags                    = var.network_subnet_az_c_tags

  internet_gateway_id   = var.network_internet_gateway_id
  internet_gateway_tags = var.network_internet_gateway_tags

  main_route_table_id      = var.network_main_route_table_id
  default_route_cidr_block = var.network_default_route_cidr_block
  main_route_table_tags    = var.network_main_route_table_tags
}

module "ec2" {
  source = "../../modules/ec2"

  enabled = var.enable_ec2

  ami_id                    = var.ec2_ami_id
  instance_type             = var.ec2_instance_type
  subnet_id                 = var.ec2_subnet_id
  vpc_security_group_ids    = local.ec2_security_group_ids_effective
  key_name                  = var.ec2_key_name
  iam_instance_profile_name = var.ec2_iam_instance_profile_name
  monitoring                = var.ec2_monitoring
  ebs_optimized             = var.ec2_ebs_optimized

  metadata_http_endpoint               = var.ec2_metadata_http_endpoint
  metadata_http_tokens                 = var.ec2_metadata_http_tokens
  metadata_http_put_response_hop_limit = var.ec2_metadata_http_put_response_hop_limit

  root_block_device_delete_on_termination = var.ec2_root_block_device_delete_on_termination
  root_block_device_volume_size           = var.ec2_root_block_device_volume_size
  root_block_device_volume_type           = var.ec2_root_block_device_volume_type
  root_block_device_iops                  = var.ec2_root_block_device_iops
  root_block_device_throughput            = var.ec2_root_block_device_throughput

  tags = var.ec2_tags
}

module "rds" {
  source = "../../modules/rds"

  enabled                = var.enable_rds
  db_instance_identifier = var.rds_instance_identifier

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
  storage_encrypted     = var.rds_storage_encrypted
  kms_key_id            = var.rds_kms_key_id

  db_subnet_group_name   = var.rds_db_subnet_group_name
  vpc_security_group_ids = local.rds_security_group_ids_effective
  parameter_group_name   = var.rds_parameter_group_name

  publicly_accessible                 = var.rds_publicly_accessible
  multi_az                            = var.rds_multi_az
  auto_minor_version_upgrade          = var.rds_auto_minor_version_upgrade
  backup_retention_period             = var.rds_backup_retention_period
  backup_window                       = var.rds_backup_window
  maintenance_window                  = var.rds_maintenance_window
  copy_tags_to_snapshot               = var.rds_copy_tags_to_snapshot
  deletion_protection                 = var.rds_deletion_protection
  monitoring_interval                 = var.rds_monitoring_interval
  performance_insights_enabled        = var.rds_performance_insights_enabled
  iam_database_authentication_enabled = var.rds_iam_database_authentication_enabled
  ca_cert_identifier                  = var.rds_ca_cert_identifier
  network_type                        = var.rds_network_type
  apply_immediately                   = var.rds_apply_immediately
  skip_final_snapshot                 = var.rds_skip_final_snapshot

  tags = var.rds_tags
}

module "elasticache" {
  source = "../../modules/elasticache"

  enabled = var.enable_elasticache

  serverless_cache_name = var.elasticache_serverless_cache_name
  engine                = var.elasticache_engine
  major_engine_version  = var.elasticache_major_engine_version
  description           = var.elasticache_description
  kms_key_id            = var.elasticache_kms_key_id
  user_group_id         = var.elasticache_user_group_id

  security_group_ids       = local.elasticache_security_group_ids_effective
  subnet_ids               = var.elasticache_subnet_ids
  snapshot_retention_limit = var.elasticache_snapshot_retention_limit
  daily_snapshot_time      = var.elasticache_daily_snapshot_time

  tags = var.elasticache_tags
}

module "acm" {
  source = "../../modules/acm"

  enabled = var.enable_acm

  domain_name               = var.acm_certificate_domain_name
  subject_alternative_names = var.acm_certificate_subject_alternative_names
  validation_method         = var.acm_validation_method
  key_algorithm             = var.acm_key_algorithm

  certificate_transparency_logging_preference = var.acm_certificate_transparency_logging_preference
  tags                                        = var.acm_tags
}

module "eks" {
  source = "../../modules/eks"

  enabled      = var.enable_eks
  tags         = var.eks_tags
  cluster_tags = var.eks_cluster_tags

  cluster_name                          = var.eks_cluster_name
  cluster_role_arn                      = local.eks_cluster_role_arn_effective
  cluster_version                       = var.eks_cluster_version
  cluster_bootstrap_self_managed_addons = var.eks_cluster_bootstrap_self_managed_addons
  cluster_subnet_ids                    = var.eks_cluster_subnet_ids
  cluster_endpoint_public_access        = var.eks_cluster_endpoint_public_access
  cluster_endpoint_private_access       = var.eks_cluster_endpoint_private_access
  cluster_public_access_cidrs           = var.eks_cluster_public_access_cidrs
  cluster_authentication_mode           = var.eks_cluster_authentication_mode
  cluster_bootstrap_admin_permissions   = var.eks_cluster_bootstrap_admin_permissions
  cluster_ip_family                     = var.eks_cluster_ip_family
  cluster_upgrade_support_type          = var.eks_cluster_upgrade_support_type
  cluster_zonal_shift_enabled           = var.eks_cluster_zonal_shift_enabled
  cluster_enabled_log_types             = var.eks_cluster_enabled_log_types

  nodegroup_enabled                = var.enable_eks_nodegroup
  nodegroup_name                   = var.eks_nodegroup_name
  nodegroup_role_arn               = local.eks_node_role_arn_effective
  nodegroup_subnet_ids             = var.eks_nodegroup_subnet_ids
  nodegroup_capacity_type          = var.eks_nodegroup_capacity_type
  nodegroup_ami_type               = var.eks_nodegroup_ami_type
  nodegroup_instance_types         = var.eks_nodegroup_instance_types
  nodegroup_disk_size              = var.eks_nodegroup_disk_size
  nodegroup_min_size               = var.eks_nodegroup_min_size
  nodegroup_max_size               = var.eks_nodegroup_max_size
  nodegroup_desired_size           = var.eks_nodegroup_desired_size
  nodegroup_update_max_unavailable = var.eks_nodegroup_update_max_unavailable
  nodegroup_repair_enabled         = var.eks_nodegroup_repair_enabled

  enable_ebs_csi_addon                       = var.enable_eks_ebs_csi_addon
  ebs_csi_addon_version                      = var.eks_ebs_csi_addon_version
  enable_cloudwatch_observability_addon      = var.enable_eks_cloudwatch_observability_addon
  cloudwatch_observability_addon_version     = var.eks_cloudwatch_observability_addon_version
  enable_ebs_csi_pod_identity_association    = var.enable_eks_ebs_csi_pod_identity_association
  ebs_csi_pod_identity_namespace             = var.eks_ebs_csi_pod_identity_namespace
  ebs_csi_pod_identity_service_account       = var.eks_ebs_csi_pod_identity_service_account
  ebs_csi_pod_identity_role_arn              = local.eks_ebs_csi_role_arn_effective
  enable_cloudwatch_pod_identity_association = var.enable_eks_cloudwatch_pod_identity_association
  cloudwatch_pod_identity_namespace          = var.eks_cloudwatch_pod_identity_namespace
  cloudwatch_pod_identity_service_account    = var.eks_cloudwatch_pod_identity_service_account
  cloudwatch_pod_identity_role_arn           = local.eks_cw_role_arn_effective
}
