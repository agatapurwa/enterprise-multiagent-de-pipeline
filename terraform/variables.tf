variable "project_id" {
  description = "Target Google Cloud Platform (GCP) project identifier."
  type        = string
  default     = "enterprise-multiagent-de-dev"
}

variable "region" {
  description = "Primary GCP deployment region for Data Lake and BigQuery resources."
  type        = string
  default     = "asia-southeast2"
}

variable "environment" {
  description = "Deployment lifecycle tier (dev, staging, or prod)."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, or prod."
  }
}

variable "bucket_name_prefix" {
  description = "Globally unique prefix for the raw ingestion and quarantine GCS Data Lake bucket."
  type        = string
  default     = "enterprise-de-lake"
}

variable "bq_dataset_id" {
  description = "Target BigQuery dataset ID for validated staging tables and agentic DDL migrations."
  type        = string
  default     = "enterprise_de_staging"
}

variable "quarantine_retention_days" {
  description = "Number of days before live anomalous/rejected files in quarantine/ are soft-deleted."
  type        = number
  default     = 30

  validation {
    condition     = var.quarantine_retention_days >= 1 && var.quarantine_retention_days <= 365
    error_message = "Quarantine retention must be between 1 and 365 days."
  }
}

variable "noncurrent_version_retention_days" {
  description = "Days after becoming noncurrent before archived GCS object versions are permanently purged."
  type        = number
  default     = 7
}