output "bucket_id" {
  description = "Backup bucket ID."
  value       = try(aws_s3_bucket.this[0].id, null)
}

output "bucket_arn" {
  description = "Backup bucket ARN."
  value       = try(aws_s3_bucket.this[0].arn, null)
}
