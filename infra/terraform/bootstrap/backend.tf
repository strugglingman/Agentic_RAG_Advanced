terraform {
  backend "s3" {
    bucket       = "agentic-rag-tfstate-543035741679-eu-north-1"
    key          = "bootstrap/terraform.tfstate"
    region       = "eu-north-1"
    encrypt      = true
    use_lockfile = true
  }
}
