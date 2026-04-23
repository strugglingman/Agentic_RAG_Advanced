locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
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
