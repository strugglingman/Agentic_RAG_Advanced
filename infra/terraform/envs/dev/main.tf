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
