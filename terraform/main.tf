terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0.0"
    }
  }

  # Remote state target (ignored locally when running `terraform init -backend=false`)
  backend "gcs" {
    prefix = "terraform/state/enterprise-multiagent-de"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  common_labels = {
    project     = "enterprise-multiagent-de-pipeline"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# =========================================================================
# 1. GCS DATA LAKE & DEAD-LETTER QUARANTINE BUCKET
# =========================================================================
resource "google_storage_bucket" "data_lake" {
  name                        = "${var.bucket_name_prefix}-${var.project_id}-${var.environment}"
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.environment != "prod"

  versioning {
    enabled = true
  }

  # Rule 1: Soft-delete live objects in quarantine/ older than retention threshold
  lifecycle_rule {
    condition {
      age            = var.quarantine_retention_days
      matches_prefix = ["quarantine/"]
    }
    action {
      type = "Delete"
    }
  }

  # Rule 2: Permanently purge noncurrent (archived) versions to prevent hidden storage costs
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = var.noncurrent_version_retention_days
      with_state                 = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  labels = local.common_labels
}

# =========================================================================
# 2. BIGQUERY STAGING & ANALYTICS DATASET
# =========================================================================
resource "google_bigquery_dataset" "staging_warehouse" {
  dataset_id                 = "${var.bq_dataset_id}_${var.environment}"
  friendly_name              = "Enterprise Multi-Agent DE Staging (${upper(var.environment)})"
  description                = "Target BigQuery dataset for vector-validated data streams and agentic DDL migrations."
  location                   = var.region
  delete_contents_on_destroy = var.environment != "prod"

  labels = local.common_labels
}