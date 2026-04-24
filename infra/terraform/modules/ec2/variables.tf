variable "enabled" {
  description = "Whether to manage the EC2 instance."
  type        = bool
  default     = false
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type."
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID where the instance is placed."
  type        = string
}

variable "vpc_security_group_ids" {
  description = "Security groups attached to the instance."
  type        = list(string)
}

variable "key_name" {
  description = "Key pair name for SSH access."
  type        = string
}

variable "iam_instance_profile_name" {
  description = "IAM instance profile name attached to EC2."
  type        = string
}

variable "monitoring" {
  description = "Whether detailed monitoring is enabled."
  type        = bool
  default     = false
}

variable "ebs_optimized" {
  description = "Whether EBS optimization is enabled."
  type        = bool
  default     = true
}

variable "metadata_http_endpoint" {
  description = "IMDS endpoint setting."
  type        = string
  default     = "enabled"
}

variable "metadata_http_tokens" {
  description = "IMDSv2 token requirement."
  type        = string
  default     = "required"
}

variable "metadata_http_put_response_hop_limit" {
  description = "IMDS hop limit."
  type        = number
  default     = 2
}

variable "root_block_device_delete_on_termination" {
  description = "Whether root volume is deleted on instance termination."
  type        = bool
  default     = true
}

variable "root_block_device_volume_size" {
  description = "Root volume size (GiB)."
  type        = number
  default     = 30
}

variable "root_block_device_volume_type" {
  description = "Root volume type."
  type        = string
  default     = "gp3"
}

variable "root_block_device_iops" {
  description = "Root volume IOPS."
  type        = number
  default     = 3000
}

variable "root_block_device_throughput" {
  description = "Root volume throughput in MiB/s."
  type        = number
  default     = 125
}

variable "tags" {
  description = "Tags applied to the EC2 instance."
  type        = map(string)
  default = {
    Name = "agentic-rag"
  }
}
