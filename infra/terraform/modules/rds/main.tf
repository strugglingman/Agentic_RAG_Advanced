resource "aws_db_instance" "this" {
  count = var.enabled ? 1 : 0

  identifier = var.db_instance_identifier

  engine         = var.engine
  engine_version = var.engine_version
  instance_class = var.instance_class

  db_name  = var.db_name
  username = var.username
  password = var.password
  port     = var.port

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = var.storage_type
  iops                  = var.iops
  storage_throughput    = var.storage_throughput
  storage_encrypted     = var.storage_encrypted
  kms_key_id            = var.kms_key_id

  db_subnet_group_name   = var.db_subnet_group_name
  vpc_security_group_ids = var.vpc_security_group_ids
  parameter_group_name   = var.parameter_group_name

  publicly_accessible                 = var.publicly_accessible
  multi_az                            = var.multi_az
  auto_minor_version_upgrade          = var.auto_minor_version_upgrade
  backup_retention_period             = var.backup_retention_period
  backup_window                       = var.backup_window
  maintenance_window                  = var.maintenance_window
  copy_tags_to_snapshot               = var.copy_tags_to_snapshot
  deletion_protection                 = var.deletion_protection
  monitoring_interval                 = var.monitoring_interval
  performance_insights_enabled        = var.performance_insights_enabled
  iam_database_authentication_enabled = var.iam_database_authentication_enabled
  ca_cert_identifier                  = var.ca_cert_identifier
  network_type                        = var.network_type
  apply_immediately                   = var.apply_immediately
  skip_final_snapshot                 = var.skip_final_snapshot

  tags = length(var.tags) > 0 ? var.tags : null

  lifecycle {
    # AWS does not return current master password; ignore to avoid fake drift.
    ignore_changes = [password]
  }
}
