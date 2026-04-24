output "acm_certificate_arn" {
  description = "ACM certificate ARN for the dev environment."
  value       = module.acm.certificate_arn
}

output "acm_dns_validation_records" {
  description = "DNS records to configure in Porkbun (or Route53) for ACM validation."
  value       = module.acm.dns_validation_records
}
