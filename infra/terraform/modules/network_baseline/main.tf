resource "aws_default_vpc" "this" {
  count = var.enabled ? 1 : 0

  enable_dns_support   = var.vpc_enable_dns_support
  enable_dns_hostnames = var.vpc_enable_dns_hostnames
  tags                 = length(var.vpc_tags) > 0 ? var.vpc_tags : null
}

resource "aws_default_subnet" "az_a" {
  count = var.enabled ? 1 : 0

  availability_zone       = var.subnet_az_a
  map_public_ip_on_launch = var.subnet_az_a_map_public_ip_on_launch
  tags                    = length(var.subnet_az_a_tags) > 0 ? var.subnet_az_a_tags : null
}

resource "aws_default_subnet" "az_b" {
  count = var.enabled ? 1 : 0

  availability_zone       = var.subnet_az_b
  map_public_ip_on_launch = var.subnet_az_b_map_public_ip_on_launch
  tags                    = length(var.subnet_az_b_tags) > 0 ? var.subnet_az_b_tags : null
}

resource "aws_default_subnet" "az_c" {
  count = var.enabled ? 1 : 0

  availability_zone       = var.subnet_az_c
  map_public_ip_on_launch = var.subnet_az_c_map_public_ip_on_launch
  tags                    = length(var.subnet_az_c_tags) > 0 ? var.subnet_az_c_tags : null
}

resource "aws_internet_gateway" "this" {
  count = var.enabled ? 1 : 0

  vpc_id = var.vpc_id
  tags   = length(var.internet_gateway_tags) > 0 ? var.internet_gateway_tags : null
}

resource "aws_route_table" "main" {
  count = var.enabled ? 1 : 0

  vpc_id = var.vpc_id

  route {
    cidr_block = var.default_route_cidr_block
    gateway_id = var.internet_gateway_id
  }

  tags = length(var.main_route_table_tags) > 0 ? var.main_route_table_tags : null
}
