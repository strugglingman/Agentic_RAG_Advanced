variable "enabled" {
  description = "Whether to manage imported security groups."
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "VPC ID containing the managed security groups."
  type        = string
}

variable "ec2_name" {
  description = "EC2 security group name."
  type        = string
}

variable "ec2_description" {
  description = "EC2 security group description."
  type        = string
}

variable "ec2_tags" {
  description = "Tags for EC2 security group."
  type        = map(string)
  default     = {}
}

variable "ec2_ingress_rules" {
  description = "Ingress rules for EC2 security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
}

variable "ec2_egress_rules" {
  description = "Egress rules for EC2 security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
}

variable "rds_name" {
  description = "RDS security group name."
  type        = string
}

variable "rds_description" {
  description = "RDS security group description."
  type        = string
}

variable "rds_tags" {
  description = "Tags for RDS security group."
  type        = map(string)
  default     = {}
}

variable "rds_ingress_rules" {
  description = "Ingress rules for RDS security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
}

variable "rds_egress_rules" {
  description = "Egress rules for RDS security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
}

variable "elasticache_name" {
  description = "ElastiCache security group name."
  type        = string
}

variable "elasticache_description" {
  description = "ElastiCache security group description."
  type        = string
}

variable "elasticache_tags" {
  description = "Tags for ElastiCache security group."
  type        = map(string)
  default     = {}
}

variable "elasticache_ingress_rules" {
  description = "Ingress rules for ElastiCache security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
}

variable "elasticache_egress_rules" {
  description = "Egress rules for ElastiCache security group."
  type = list(object({
    description      = string
    from_port        = number
    to_port          = number
    protocol         = string
    cidr_blocks      = list(string)
    ipv6_cidr_blocks = list(string)
    prefix_list_ids  = list(string)
    security_groups  = list(string)
    self             = bool
  }))
}
