data "aws_iam_policy_document" "eks_cluster_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "eks_pod_identity_assume_role" {
  statement {
    effect = "Allow"
    actions = [
      "sts:AssumeRole",
      "sts:TagSession",
    ]

    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "s3_backup" {
  statement {
    sid     = "ListBackupBucket"
    effect  = "Allow"
    actions = ["s3:ListBucket"]

    resources = [
      "arn:aws:s3:::${var.s3_backup_bucket_name}",
    ]
  }

  statement {
    sid    = "ManageBackupObjects"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:DeleteObject",
    ]

    resources = [
      "arn:aws:s3:::${var.s3_backup_bucket_name}/*",
    ]
  }
}

data "aws_iam_policy_document" "ecr_push" {
  statement {
    sid       = "AllowECRAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "AllowPushToAgenticRagRepos"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeRepositories",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:ListImages",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]

    resources = [
      "arn:aws:ecr:${var.aws_region}:${var.aws_account_id}:repository/${var.ecr_backend_repo_name}",
      "arn:aws:ecr:${var.aws_region}:${var.aws_account_id}:repository/${var.ecr_frontend_repo_name}",
    ]
  }
}

locals {
  base_role_definitions = {
    eks_cluster = {
      name               = var.eks_cluster_role_name
      assume_role_policy = data.aws_iam_policy_document.eks_cluster_assume_role.json
      description        = null
    }
    eks_node = {
      name               = var.eks_node_role_name
      assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
      description        = null
    }
    ec2_backup = {
      name               = var.ec2_backup_role_name
      assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
      description        = var.ec2_backup_role_description
    }
  }

  optional_role_definitions = merge(
    var.enable_eks_ebs_csi_role ? {
      eks_ebs_csi = {
        name               = var.eks_ebs_csi_role_name
        assume_role_policy = data.aws_iam_policy_document.eks_pod_identity_assume_role.json
        description        = null
      }
    } : {},
    var.enable_eks_cloudwatch_observability_role ? {
      eks_cloudwatch_observability = {
        name               = var.eks_cloudwatch_observability_role_name
        assume_role_policy = data.aws_iam_policy_document.eks_pod_identity_assume_role.json
        description        = null
      }
    } : {}
  )

  role_definitions = merge(local.base_role_definitions, local.optional_role_definitions)

  base_aws_managed_policy_attachments = {
    eks_cluster_AmazonEKSClusterPolicy = {
      role_key   = "eks_cluster"
      policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
    }
    eks_node_AmazonEKSWorkerNodePolicy = {
      role_key   = "eks_node"
      policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
    }
    eks_node_AmazonEC2ContainerRegistryPullOnly = {
      role_key   = "eks_node"
      policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly"
    }
    eks_node_AmazonEKS_CNI_Policy = {
      role_key   = "eks_node"
      policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
    }
  }

  optional_aws_managed_policy_attachments = merge(
    var.enable_eks_ebs_csi_role ? {
      eks_ebs_csi_AmazonEBSCSIDriverPolicy = {
        role_key   = "eks_ebs_csi"
        policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
      }
    } : {},
    var.enable_eks_cloudwatch_observability_role ? {
      eks_cloudwatch_observability_CloudWatchAgentServerPolicy = {
        role_key   = "eks_cloudwatch_observability"
        policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
      }
    } : {}
  )

  aws_managed_policy_attachments = merge(local.base_aws_managed_policy_attachments, local.optional_aws_managed_policy_attachments)
}

resource "aws_iam_role" "this" {
  for_each = var.enabled ? local.role_definitions : {}

  name               = each.value.name
  assume_role_policy = each.value.assume_role_policy
  description        = each.value.description
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "aws_managed" {
  for_each = var.enabled ? local.aws_managed_policy_attachments : {}

  role       = aws_iam_role.this[each.value.role_key].name
  policy_arn = each.value.policy_arn
}

resource "aws_iam_policy" "s3_backup" {
  count = var.enabled ? 1 : 0

  name   = var.s3_backup_policy_name
  description = var.s3_backup_policy_description
  policy = data.aws_iam_policy_document.s3_backup.json
  tags   = var.tags
}

resource "aws_iam_policy" "ecr_push" {
  count = var.enabled ? 1 : 0

  name   = var.ecr_push_policy_name
  description = var.ecr_push_policy_description
  policy = data.aws_iam_policy_document.ecr_push.json
  tags   = var.tags
}

resource "aws_iam_role_policy_attachment" "ec2_s3_backup" {
  count = var.enabled ? 1 : 0

  role       = aws_iam_role.this["ec2_backup"].name
  policy_arn = aws_iam_policy.s3_backup[0].arn
}

resource "aws_iam_role_policy_attachment" "ec2_ecr_push" {
  count = var.enabled ? 1 : 0

  role       = aws_iam_role.this["ec2_backup"].name
  policy_arn = aws_iam_policy.ecr_push[0].arn
}

resource "aws_iam_instance_profile" "ec2_backup" {
  count = var.enabled ? 1 : 0

  name = var.ec2_instance_profile_name
  role = aws_iam_role.this["ec2_backup"].name
  tags = var.tags
}
