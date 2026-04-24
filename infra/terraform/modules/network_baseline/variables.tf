variable "enabled" {
  description = "Whether to manage imported baseline network resources for the default VPC."
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "Default VPC ID."
  type        = string
}

variable "vpc_enable_dns_support" {
  description = "Whether DNS resolution is enabled in the VPC."
  type        = bool
  default     = true
}

variable "vpc_enable_dns_hostnames" {
  description = "Whether DNS hostnames are enabled in the VPC."
  type        = bool
  default     = true
}

variable "vpc_tags" {
  description = "Tags for default VPC."
  type        = map(string)
  default     = {}
}

variable "subnet_az_a" {
  description = "Availability Zone name for default subnet A."
  type        = string
}

variable "subnet_az_a_map_public_ip_on_launch" {
  description = "Whether subnet A maps public IP on launch."
  type        = bool
  default     = true
}

variable "subnet_az_a_tags" {
  description = "Tags for default subnet A."
  type        = map(string)
  default     = {}
}

variable "subnet_az_b" {
  description = "Availability Zone name for default subnet B."
  type        = string
}

variable "subnet_az_b_map_public_ip_on_launch" {
  description = "Whether subnet B maps public IP on launch."
  type        = bool
  default     = true
}

variable "subnet_az_b_tags" {
  description = "Tags for default subnet B."
  type        = map(string)
  default     = {}
}

variable "subnet_az_c" {
  description = "Availability Zone name for default subnet C."
  type        = string
}

variable "subnet_az_c_map_public_ip_on_launch" {
  description = "Whether subnet C maps public IP on launch."
  type        = bool
  default     = true
}

variable "subnet_az_c_tags" {
  description = "Tags for default subnet C."
  type        = map(string)
  default     = {}
}

variable "internet_gateway_id" {
  description = "Internet gateway ID attached to the default VPC."
  type        = string
}

variable "internet_gateway_tags" {
  description = "Tags for internet gateway."
  type        = map(string)
  default     = {}
}

variable "main_route_table_id" {
  description = "Main route table ID in the default VPC."
  type        = string
}

variable "default_route_cidr_block" {
  description = "Default route destination CIDR block for internet egress."
  type        = string
  default     = "0.0.0.0/0"
}

variable "main_route_table_tags" {
  description = "Tags for main route table."
  type        = map(string)
  default     = {}
}
