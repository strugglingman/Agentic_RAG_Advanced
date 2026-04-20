output "sns_topic_arn" {
  description = "SNS topic ARN for alerts."
  value       = var.enabled ? aws_sns_topic.alerts[0].arn : null
}
