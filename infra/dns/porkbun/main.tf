data "terraform_remote_state" "aws_env" {
  count   = var.manage_acm_validation_records || var.enable_acm_certificate_validation ? 1 : 0
  backend = "s3"

  config = {
    bucket = var.aws_state_bucket
    key    = var.aws_state_key
    region = var.aws_state_region
  }
}

locals {
  domain_normalized = lower(trimsuffix(var.domain, "."))

  acm_certificate_arn = try(data.terraform_remote_state.aws_env[0].outputs.acm_certificate_arn, null)
  acm_dns_records_raw = try(data.terraform_remote_state.aws_env[0].outputs.acm_dns_validation_records, [])

  # Porkbun expects relative record names (subdomain part only).
  acm_dns_records = {
    for rec in local.acm_dns_records_raw : lower(trimsuffix(rec.resource_record_name, ".")) => {
      name = trimsuffix(
        lower(trimsuffix(rec.resource_record_name, ".")),
        ".${local.domain_normalized}"
      )
      type    = upper(rec.resource_record_type)
      content = trimsuffix(rec.resource_record_value, ".")
      fqdn    = lower(trimsuffix(rec.resource_record_name, "."))
    }
  }

  app_dns_records = {
    for rec in var.app_dns_records : "${upper(rec.type)}:${lower(rec.name)}" => {
      name = rec.name == "" || rec.name == "@" ? "" : trimsuffix(
        lower(trimsuffix(rec.name, ".")),
        ".${local.domain_normalized}"
      )
      type     = upper(rec.type)
      content  = trimsuffix(rec.content, ".")
      ttl      = rec.ttl
      priority = try(rec.priority, null)
      notes    = try(rec.notes, null)
    }
  }

  acm_validation_record_fqdns = [for key in sort(keys(local.acm_dns_records)) : local.acm_dns_records[key].fqdn]
}

resource "porkbun_dns_record" "acm_validation" {
  for_each = var.manage_acm_validation_records ? local.acm_dns_records : {}

  domain  = local.domain_normalized
  name    = each.value.name
  type    = each.value.type
  content = each.value.content
  ttl     = var.acm_validation_ttl
  notes   = "Managed by Terraform for ACM DNS validation"
}

resource "porkbun_dns_record" "app" {
  for_each = local.app_dns_records

  domain   = local.domain_normalized
  name     = each.value.name
  type     = each.value.type
  content  = each.value.content
  ttl      = each.value.ttl
  priority = each.value.priority
  notes    = each.value.notes
}

resource "aws_acm_certificate_validation" "this" {
  count = var.enable_acm_certificate_validation ? 1 : 0

  certificate_arn         = local.acm_certificate_arn
  validation_record_fqdns = local.acm_validation_record_fqdns

  depends_on = [
    porkbun_dns_record.acm_validation,
  ]

  lifecycle {
    precondition {
      condition     = local.acm_certificate_arn != null && local.acm_certificate_arn != ""
      error_message = "acm_certificate_arn is empty in remote state. Apply infra/terraform/envs/dev with ACM enabled first."
    }
    precondition {
      condition     = length(local.acm_validation_record_fqdns) > 0
      error_message = "acm_dns_validation_records is empty in remote state. ACM DNS validation records must exist before validation."
    }
  }
}
