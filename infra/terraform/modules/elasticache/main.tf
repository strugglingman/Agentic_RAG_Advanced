resource "aws_elasticache_serverless_cache" "this" {
  count = var.enabled ? 1 : 0

  name = var.serverless_cache_name

  engine               = var.engine
  major_engine_version = var.major_engine_version
  description          = var.description
  kms_key_id           = var.kms_key_id
  user_group_id        = var.user_group_id

  security_group_ids       = var.security_group_ids
  subnet_ids               = var.subnet_ids
  snapshot_retention_limit = var.snapshot_retention_limit
  daily_snapshot_time      = var.daily_snapshot_time

  tags = length(var.tags) > 0 ? var.tags : null
}
