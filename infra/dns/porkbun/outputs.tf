output "acm_validation_record_fqdns" {
  description = "FQDNs used by ACM certificate validation."
  value       = local.acm_validation_record_fqdns
}

output "acm_dns_records_applied" {
  description = "ACM DNS validation records managed in Porkbun."
  value = {
    for key, record in porkbun_dns_record.acm_validation : key => {
      id      = record.id
      domain  = record.domain
      name    = record.name
      type    = record.type
      content = record.content
      ttl     = record.ttl
    }
  }
}

output "app_dns_records_applied" {
  description = "Application DNS records managed in Porkbun."
  value = {
    for key, record in porkbun_dns_record.app : key => {
      id      = record.id
      domain  = record.domain
      name    = record.name
      type    = record.type
      content = record.content
      ttl     = record.ttl
    }
  }
}

output "acm_certificate_validation_id" {
  description = "ACM certificate validation resource ID timestamp."
  value       = try(aws_acm_certificate_validation.this[0].id, null)
}
