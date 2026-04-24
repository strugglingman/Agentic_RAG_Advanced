output "serverless_cache_id" {
  description = "ElastiCache serverless cache ID."
  value       = try(aws_elasticache_serverless_cache.this[0].id, null)
}

output "serverless_cache_arn" {
  description = "ElastiCache serverless cache ARN."
  value       = try(aws_elasticache_serverless_cache.this[0].arn, null)
}

output "endpoint" {
  description = "Primary endpoint for serverless cache."
  value       = try(aws_elasticache_serverless_cache.this[0].endpoint, null)
}
