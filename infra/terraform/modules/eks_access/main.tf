locals {
  entry_map = var.enabled ? var.access_entries : {}

  policy_association_map = {
    for assoc in flatten([
      for entry_key, entry in local.entry_map : [
        for idx, policy in entry.policy_associations : {
          key               = "${entry_key}-${idx}"
          entry_key         = entry_key
          principal_arn     = entry.principal_arn
          policy_arn        = policy.policy_arn
          access_scope_type = policy.access_scope_type
          namespaces        = policy.namespaces
        }
      ]
    ]) : assoc.key => assoc
  }
}

resource "aws_eks_access_entry" "this" {
  for_each = local.entry_map

  cluster_name      = var.cluster_name
  principal_arn     = each.value.principal_arn
  type              = each.value.type
  kubernetes_groups = each.value.kubernetes_groups
  user_name         = each.value.username
  tags              = length(each.value.tags) > 0 ? each.value.tags : null
}

resource "aws_eks_access_policy_association" "this" {
  for_each = local.policy_association_map

  cluster_name  = var.cluster_name
  principal_arn = each.value.principal_arn
  policy_arn    = each.value.policy_arn

  access_scope {
    type       = each.value.access_scope_type
    namespaces = each.value.access_scope_type == "namespace" ? each.value.namespaces : null
  }

  depends_on = [aws_eks_access_entry.this]
}
