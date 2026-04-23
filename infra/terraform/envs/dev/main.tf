locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  eks_cluster_role_arn_effective = var.eks_cluster_role_arn != "" ? var.eks_cluster_role_arn : try(module.iam.eks_cluster_role_arn, "")
  eks_node_role_arn_effective    = var.eks_node_role_arn != "" ? var.eks_node_role_arn : try(module.iam.eks_node_role_arn, "")
  eks_ebs_csi_role_arn_effective = var.eks_ebs_csi_role_arn != "" ? var.eks_ebs_csi_role_arn : try(module.iam.eks_ebs_csi_role_arn, "")
  eks_cw_role_arn_effective      = var.eks_cloudwatch_observability_role_arn != "" ? var.eks_cloudwatch_observability_role_arn : try(module.iam.eks_cloudwatch_observability_role_arn, "")
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
