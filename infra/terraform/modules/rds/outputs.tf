output "db_instance_id" {
  description = "RDS instance identifier."
  value       = try(aws_db_instance.this[0].id, null)
}

output "db_instance_arn" {
  description = "RDS instance ARN."
  value       = try(aws_db_instance.this[0].arn, null)
}

output "endpoint" {
  description = "RDS endpoint address."
  value       = try(aws_db_instance.this[0].address, null)
}
