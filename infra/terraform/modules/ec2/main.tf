resource "aws_instance" "this" {
  count = var.enabled ? 1 : 0

  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = var.vpc_security_group_ids
  key_name               = var.key_name
  iam_instance_profile   = var.iam_instance_profile_name
  monitoring             = var.monitoring
  ebs_optimized          = var.ebs_optimized

  metadata_options {
    http_endpoint               = var.metadata_http_endpoint
    http_tokens                 = var.metadata_http_tokens
    http_put_response_hop_limit = var.metadata_http_put_response_hop_limit
  }

  root_block_device {
    delete_on_termination = var.root_block_device_delete_on_termination
    volume_size           = var.root_block_device_volume_size
    volume_type           = var.root_block_device_volume_type
    iops                  = var.root_block_device_iops
    throughput            = var.root_block_device_throughput
  }

  tags = var.tags
}
