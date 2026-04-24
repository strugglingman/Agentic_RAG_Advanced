output "certificate_arn" {
  description = "ACM certificate ARN."
  value       = try(aws_acm_certificate.this[0].arn, null)
}

output "dns_validation_records" {
  description = "DNS validation records required for ACM domain validation."
  value = try(
    [
      for dvo in aws_acm_certificate.this[0].domain_validation_options : {
        domain_name           = dvo.domain_name
        resource_record_name  = dvo.resource_record_name
        resource_record_type  = dvo.resource_record_type
        resource_record_value = dvo.resource_record_value
      }
    ],
    []
  )
}
