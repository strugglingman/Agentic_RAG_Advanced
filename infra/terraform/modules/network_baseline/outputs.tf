output "vpc_id" {
  description = "Default VPC ID."
  value       = try(aws_default_vpc.this[0].id, null)
}

output "subnet_ids" {
  description = "Managed default subnet IDs."
  value = compact([
    try(aws_default_subnet.az_a[0].id, null),
    try(aws_default_subnet.az_b[0].id, null),
    try(aws_default_subnet.az_c[0].id, null),
  ])
}

output "internet_gateway_id" {
  description = "Internet gateway ID."
  value       = try(aws_internet_gateway.this[0].id, null)
}

output "main_route_table_id" {
  description = "Main route table ID."
  value       = try(aws_route_table.main[0].id, null)
}
