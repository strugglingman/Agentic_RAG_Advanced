resource "aws_security_group" "ec2" {
  count = var.enabled ? 1 : 0

  name        = var.ec2_name
  description = var.ec2_description
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = var.ec2_ingress_rules
    content {
      description      = ingress.value.description != "" ? ingress.value.description : null
      from_port        = ingress.value.from_port
      to_port          = ingress.value.to_port
      protocol         = ingress.value.protocol
      cidr_blocks      = ingress.value.cidr_blocks
      ipv6_cidr_blocks = ingress.value.ipv6_cidr_blocks
      prefix_list_ids  = ingress.value.prefix_list_ids
      security_groups  = ingress.value.security_groups
      self             = ingress.value.self
    }
  }

  dynamic "egress" {
    for_each = var.ec2_egress_rules
    content {
      description      = egress.value.description != "" ? egress.value.description : null
      from_port        = egress.value.from_port
      to_port          = egress.value.to_port
      protocol         = egress.value.protocol
      cidr_blocks      = egress.value.cidr_blocks
      ipv6_cidr_blocks = egress.value.ipv6_cidr_blocks
      prefix_list_ids  = egress.value.prefix_list_ids
      security_groups  = egress.value.security_groups
      self             = egress.value.self
    }
  }

  tags = length(var.ec2_tags) > 0 ? var.ec2_tags : null
}

resource "aws_security_group" "rds" {
  count = var.enabled ? 1 : 0

  name        = var.rds_name
  description = var.rds_description
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = var.rds_ingress_rules
    content {
      description      = ingress.value.description != "" ? ingress.value.description : null
      from_port        = ingress.value.from_port
      to_port          = ingress.value.to_port
      protocol         = ingress.value.protocol
      cidr_blocks      = ingress.value.cidr_blocks
      ipv6_cidr_blocks = ingress.value.ipv6_cidr_blocks
      prefix_list_ids  = ingress.value.prefix_list_ids
      security_groups  = ingress.value.security_groups
      self             = ingress.value.self
    }
  }

  dynamic "egress" {
    for_each = var.rds_egress_rules
    content {
      description      = egress.value.description != "" ? egress.value.description : null
      from_port        = egress.value.from_port
      to_port          = egress.value.to_port
      protocol         = egress.value.protocol
      cidr_blocks      = egress.value.cidr_blocks
      ipv6_cidr_blocks = egress.value.ipv6_cidr_blocks
      prefix_list_ids  = egress.value.prefix_list_ids
      security_groups  = egress.value.security_groups
      self             = egress.value.self
    }
  }

  tags = length(var.rds_tags) > 0 ? var.rds_tags : null
}

resource "aws_security_group" "elasticache" {
  count = var.enabled ? 1 : 0

  name        = var.elasticache_name
  description = var.elasticache_description
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = var.elasticache_ingress_rules
    content {
      description      = ingress.value.description != "" ? ingress.value.description : null
      from_port        = ingress.value.from_port
      to_port          = ingress.value.to_port
      protocol         = ingress.value.protocol
      cidr_blocks      = ingress.value.cidr_blocks
      ipv6_cidr_blocks = ingress.value.ipv6_cidr_blocks
      prefix_list_ids  = ingress.value.prefix_list_ids
      security_groups  = ingress.value.security_groups
      self             = ingress.value.self
    }
  }

  dynamic "egress" {
    for_each = var.elasticache_egress_rules
    content {
      description      = egress.value.description != "" ? egress.value.description : null
      from_port        = egress.value.from_port
      to_port          = egress.value.to_port
      protocol         = egress.value.protocol
      cidr_blocks      = egress.value.cidr_blocks
      ipv6_cidr_blocks = egress.value.ipv6_cidr_blocks
      prefix_list_ids  = egress.value.prefix_list_ids
      security_groups  = egress.value.security_groups
      self             = egress.value.self
    }
  }

  tags = length(var.elasticache_tags) > 0 ? var.elasticache_tags : null
}

resource "aws_security_group" "eks_cluster" {
  count = var.enabled && var.enable_eks_cluster_security_group ? 1 : 0

  name        = var.eks_cluster_name
  description = var.eks_cluster_description
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = var.eks_cluster_ingress_rules
    content {
      description      = ingress.value.description != "" ? ingress.value.description : null
      from_port        = ingress.value.from_port
      to_port          = ingress.value.to_port
      protocol         = ingress.value.protocol
      cidr_blocks      = ingress.value.cidr_blocks
      ipv6_cidr_blocks = ingress.value.ipv6_cidr_blocks
      prefix_list_ids  = ingress.value.prefix_list_ids
      security_groups  = ingress.value.security_groups
      self             = ingress.value.self
    }
  }

  dynamic "egress" {
    for_each = var.eks_cluster_egress_rules
    content {
      description      = egress.value.description != "" ? egress.value.description : null
      from_port        = egress.value.from_port
      to_port          = egress.value.to_port
      protocol         = egress.value.protocol
      cidr_blocks      = egress.value.cidr_blocks
      ipv6_cidr_blocks = egress.value.ipv6_cidr_blocks
      prefix_list_ids  = egress.value.prefix_list_ids
      security_groups  = egress.value.security_groups
      self             = egress.value.self
    }
  }

  tags = length(var.eks_cluster_tags) > 0 ? var.eks_cluster_tags : null
}

resource "aws_default_security_group" "default_vpc" {
  count = var.enabled && var.enable_default_vpc_security_group ? 1 : 0

  vpc_id = var.vpc_id

  dynamic "ingress" {
    for_each = var.default_vpc_ingress_rules
    content {
      description      = ingress.value.description != "" ? ingress.value.description : null
      from_port        = ingress.value.from_port
      to_port          = ingress.value.to_port
      protocol         = ingress.value.protocol
      cidr_blocks      = ingress.value.cidr_blocks
      ipv6_cidr_blocks = ingress.value.ipv6_cidr_blocks
      prefix_list_ids  = ingress.value.prefix_list_ids
      security_groups  = ingress.value.security_groups
      self             = ingress.value.self
    }
  }

  dynamic "egress" {
    for_each = var.default_vpc_egress_rules
    content {
      description      = egress.value.description != "" ? egress.value.description : null
      from_port        = egress.value.from_port
      to_port          = egress.value.to_port
      protocol         = egress.value.protocol
      cidr_blocks      = egress.value.cidr_blocks
      ipv6_cidr_blocks = egress.value.ipv6_cidr_blocks
      prefix_list_ids  = egress.value.prefix_list_ids
      security_groups  = egress.value.security_groups
      self             = egress.value.self
    }
  }

  tags = length(var.default_vpc_tags) > 0 ? var.default_vpc_tags : null
}
