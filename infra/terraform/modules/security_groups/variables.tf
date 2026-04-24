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

variable "enable_eks_cluster_security_group" {
  description = "Whether to manage the imported EKS cluster security group."
  type        = bool
  default     = false
}

variable "eks_cluster_name" {
  description = "EKS cluster security group name."
  type        = string
  default     = ""
}

variable "eks_cluster_description" {
  description = "EKS cluster security group description."
  type        = string
  default     = ""
}

variable "eks_cluster_tags" {
  description = "Tags for EKS cluster security group."
  type        = map(string)
  default     = {}
}

variable "eks_cluster_ingress_rules" {
  description = "Ingress rules for EKS cluster security group."
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
  default = []
}

variable "eks_cluster_egress_rules" {
  description = "Egress rules for EKS cluster security group."
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
  default = []
}

variable "enable_default_vpc_security_group" {
  description = "Whether to manage the default VPC security group."
  type        = bool
  default     = false
}

variable "default_vpc_ingress_rules" {
  description = "Ingress rules for default VPC security group."
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
  default = []
}

variable "default_vpc_egress_rules" {
  description = "Egress rules for default VPC security group."
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
  default = []
}

variable "default_vpc_tags" {
  description = "Tags for default VPC security group."
  type        = map(string)
  default     = {}
}
