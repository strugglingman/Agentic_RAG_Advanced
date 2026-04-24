resource "aws_ecr_repository" "backend" {
  count = var.enabled ? 1 : 0

  name                 = var.backend_repository_name
  image_tag_mutability = var.image_tag_mutability
  force_delete         = var.force_delete

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = var.encryption_type
    kms_key         = var.encryption_type == "KMS" ? var.kms_key : null
  }

  tags = length(var.tags) > 0 ? var.tags : null
}

resource "aws_ecr_repository" "frontend" {
  count = var.enabled ? 1 : 0

  name                 = var.frontend_repository_name
  image_tag_mutability = var.image_tag_mutability
  force_delete         = var.force_delete

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = var.encryption_type
    kms_key         = var.encryption_type == "KMS" ? var.kms_key : null
  }

  tags = length(var.tags) > 0 ? var.tags : null
}
