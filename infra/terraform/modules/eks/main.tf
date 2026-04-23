resource "aws_eks_cluster" "this" {
  count = var.enabled ? 1 : 0

  name     = var.cluster_name
  role_arn = var.cluster_role_arn
  version  = var.cluster_version
  bootstrap_self_managed_addons = var.cluster_bootstrap_self_managed_addons

  vpc_config {
    subnet_ids              = var.cluster_subnet_ids
    endpoint_public_access  = var.cluster_endpoint_public_access
    endpoint_private_access = var.cluster_endpoint_private_access
    public_access_cidrs     = var.cluster_public_access_cidrs
  }

  kubernetes_network_config {
    ip_family = var.cluster_ip_family
  }

  access_config {
    authentication_mode                         = var.cluster_authentication_mode
    bootstrap_cluster_creator_admin_permissions = var.cluster_bootstrap_admin_permissions
  }

  upgrade_policy {
    support_type = var.cluster_upgrade_support_type
  }

  zonal_shift_config {
    enabled = var.cluster_zonal_shift_enabled
  }

  enabled_cluster_log_types = var.cluster_enabled_log_types
  tags                      = var.cluster_tags
}

resource "aws_eks_node_group" "this" {
  count = var.enabled && var.nodegroup_enabled ? 1 : 0

  cluster_name    = aws_eks_cluster.this[0].name
  node_group_name = var.nodegroup_name
  node_role_arn   = var.nodegroup_role_arn
  subnet_ids      = var.nodegroup_subnet_ids
  capacity_type   = var.nodegroup_capacity_type
  ami_type        = var.nodegroup_ami_type
  instance_types  = var.nodegroup_instance_types
  disk_size       = var.nodegroup_disk_size

  scaling_config {
    min_size     = var.nodegroup_min_size
    max_size     = var.nodegroup_max_size
    desired_size = var.nodegroup_desired_size
  }

  update_config {
    max_unavailable = var.nodegroup_update_max_unavailable
  }

  node_repair_config {
    enabled = var.nodegroup_repair_enabled
  }

  tags = length(var.tags) > 0 ? var.tags : null
}

resource "aws_eks_addon" "ebs_csi" {
  count = var.enabled && var.enable_ebs_csi_addon ? 1 : 0

  cluster_name      = aws_eks_cluster.this[0].name
  addon_name        = "aws-ebs-csi-driver"
  addon_version     = var.ebs_csi_addon_version
  tags              = length(var.tags) > 0 ? var.tags : null

  dynamic "pod_identity_association" {
    for_each = var.enable_ebs_csi_pod_identity_association && var.ebs_csi_pod_identity_role_arn != "" ? [1] : []
    content {
      role_arn        = var.ebs_csi_pod_identity_role_arn
      service_account = var.ebs_csi_pod_identity_service_account
    }
  }
}

resource "aws_eks_addon" "cloudwatch_observability" {
  count = var.enabled && var.enable_cloudwatch_observability_addon ? 1 : 0

  cluster_name      = aws_eks_cluster.this[0].name
  addon_name        = "amazon-cloudwatch-observability"
  addon_version     = var.cloudwatch_observability_addon_version
  tags              = length(var.tags) > 0 ? var.tags : null

  dynamic "pod_identity_association" {
    for_each = var.enable_cloudwatch_pod_identity_association && var.cloudwatch_pod_identity_role_arn != "" ? [1] : []
    content {
      role_arn        = var.cloudwatch_pod_identity_role_arn
      service_account = var.cloudwatch_pod_identity_service_account
    }
  }
}

resource "aws_eks_pod_identity_association" "ebs_csi" {
  count = var.enabled && var.enable_ebs_csi_pod_identity_association && var.ebs_csi_pod_identity_role_arn != "" ? 1 : 0

  cluster_name    = aws_eks_cluster.this[0].name
  namespace       = var.ebs_csi_pod_identity_namespace
  service_account = var.ebs_csi_pod_identity_service_account
  role_arn        = var.ebs_csi_pod_identity_role_arn
  tags            = length(var.tags) > 0 ? var.tags : null
}

resource "aws_eks_pod_identity_association" "cloudwatch" {
  count = var.enabled && var.enable_cloudwatch_pod_identity_association && var.cloudwatch_pod_identity_role_arn != "" ? 1 : 0

  cluster_name    = aws_eks_cluster.this[0].name
  namespace       = var.cloudwatch_pod_identity_namespace
  service_account = var.cloudwatch_pod_identity_service_account
  role_arn        = var.cloudwatch_pod_identity_role_arn
  tags            = length(var.tags) > 0 ? var.tags : null
}
