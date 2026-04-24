resource "aws_sns_topic" "alerts" {
  count = var.enabled ? 1 : 0

  name = var.sns_topic_name
  tags = var.tags
}

resource "aws_sns_topic_subscription" "email" {
  count = var.enabled && var.alert_email_endpoint != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email_endpoint
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  count = var.enabled ? 1 : 0

  alarm_name          = "${var.sns_topic_name}-rds-cpu-high"
  alarm_description   = "RDS CPUUtilization average > 70 for 2/3 periods"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 70
  treat_missing_data  = "missing"

  dimensions = {
    DBInstanceIdentifier = var.rds_instance_id
  }

  alarm_actions = [aws_sns_topic.alerts[0].arn]
  ok_actions    = [aws_sns_topic.alerts[0].arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "redis_evictions_high" {
  count = var.enabled ? 1 : 0

  alarm_name          = "${var.sns_topic_name}-redis-evictions-high"
  alarm_description   = "Redis evictions > 0 in evaluation window"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Evictions"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"

  dimensions = var.redis_alarm_dimensions

  alarm_actions = [aws_sns_topic.alerts[0].arn]
  ok_actions    = [aws_sns_topic.alerts[0].arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "eks_pod_restart_high" {
  count = var.enabled ? 1 : 0

  alarm_name          = "${var.alarm_name_prefix}-eks-pod-restarts-high"
  alarm_description   = "ContainerInsights reports pod container restarts in agentic-rag namespace."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  metric_name         = "namespace_number_of_container_restarts"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"

  dimensions = {
    Namespace = var.eks_namespace
  }

  alarm_actions             = [aws_sns_topic.alerts[0].arn]
  ok_actions                = []
  insufficient_data_actions = []
  tags                      = var.tags
}

resource "aws_cloudwatch_metric_alarm" "rds_storage_low" {
  count = var.enabled ? 1 : 0

  alarm_name          = "${var.alarm_name_prefix}-rds-storage-low"
  alarm_description   = "RDS FreeStorageSpace is below 10 GiB."
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 10737418240
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = var.rds_instance_id
  }

  alarm_actions             = [aws_sns_topic.alerts[0].arn]
  ok_actions                = []
  insufficient_data_actions = []
  tags                      = var.tags
}

resource "aws_cloudwatch_metric_alarm" "redis_engine_cpu_high" {
  count = var.enabled ? 1 : 0

  alarm_name          = "${var.alarm_name_prefix}-redis-engine-cpu-high"
  alarm_description   = "ElastiCache engine CPU utilization is above baseline threshold."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  metric_name         = "EngineCPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 70
  treat_missing_data  = "notBreaching"

  dimensions = var.redis_engine_alarm_dimensions

  alarm_actions             = [aws_sns_topic.alerts[0].arn]
  ok_actions                = []
  insufficient_data_actions = []
  tags                      = var.tags
}
