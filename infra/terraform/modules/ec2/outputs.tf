output "instance_id" {
  description = "EC2 instance ID."
  value       = try(aws_instance.this[0].id, null)
}

output "instance_arn" {
  description = "EC2 instance ARN."
  value       = try(aws_instance.this[0].arn, null)
}

output "private_ip" {
  description = "Private IP of the EC2 instance."
  value       = try(aws_instance.this[0].private_ip, null)
}

output "public_ip" {
  description = "Public IP of the EC2 instance."
  value       = try(aws_instance.this[0].public_ip, null)
}
