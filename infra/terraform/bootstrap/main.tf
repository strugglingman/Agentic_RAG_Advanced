locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Week 1-2 intentionally keeps bootstrap root non-destructive.
# Remote state resources (S3 + lock table) are added in Week 2
# after naming and access decisions are finalized.
