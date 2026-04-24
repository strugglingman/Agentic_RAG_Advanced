resource "aws_s3_bucket" "this" {
  count = var.enabled ? 1 : 0

  bucket        = var.bucket_name
  force_destroy = var.force_destroy
  tags          = length(var.tags) > 0 ? var.tags : null
}

resource "aws_s3_bucket_public_access_block" "this" {
  count = var.enabled && var.manage_public_access_block ? 1 : 0

  bucket = aws_s3_bucket.this[0].id

  block_public_acls       = var.block_public_acls
  ignore_public_acls      = var.ignore_public_acls
  block_public_policy     = var.block_public_policy
  restrict_public_buckets = var.restrict_public_buckets
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  count = var.enabled && var.manage_server_side_encryption ? 1 : 0

  bucket = aws_s3_bucket.this[0].id

  rule {
    bucket_key_enabled = var.bucket_key_enabled

    apply_server_side_encryption_by_default {
      sse_algorithm     = var.sse_algorithm
      kms_master_key_id = var.sse_algorithm == "aws:kms" ? var.kms_master_key_id : null
    }
  }
}

resource "aws_s3_bucket_ownership_controls" "this" {
  count = var.enabled && var.manage_ownership_controls ? 1 : 0

  bucket = aws_s3_bucket.this[0].id

  rule {
    object_ownership = var.object_ownership
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
  count = var.enabled && var.manage_lifecycle_configuration ? 1 : 0

  bucket                                 = aws_s3_bucket.this[0].id
  transition_default_minimum_object_size = var.transition_default_minimum_object_size

  rule {
    id     = var.lifecycle_rule_id
    status = "Enabled"

    filter {
      prefix = var.lifecycle_prefix
    }

    expiration {
      days = var.lifecycle_expiration_days
    }
  }
}
