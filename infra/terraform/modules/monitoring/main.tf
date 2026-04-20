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
