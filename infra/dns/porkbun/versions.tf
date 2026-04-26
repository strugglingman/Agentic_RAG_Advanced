terraform {
  required_version = ">= 1.8.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    # Community provider for Porkbun DNS APIs.
    porkbun = {
      source  = "kyswtn/porkbun"
      version = "= 0.1.3"
    }
  }
}
