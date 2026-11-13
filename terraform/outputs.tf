output "data_lake_bucket_name" {
  description = "Provisioned GCS Data Lake and quarantine bucket name."
  value       = google_storage_bucket.data_lake.name
}

output "data_lake_bucket_url" {
  description = "Base gs:// URI for raw and quarantine storage prefixes."
  value       = google_storage_bucket.data_lake.url
}

output "quarantine_prefix_uri" {
  description = "Target dead-letter quarantine prefix URI for rejected batches."
  value       = "${google_storage_bucket.data_lake.url}/quarantine"
}

output "bigquery_dataset_id" {
  description = "Target BigQuery staging dataset ID for validated ingestion and AgenticOperator DDL."
  value       = google_bigquery_dataset.staging_warehouse.dataset_id
}

output "bigquery_dataset_full_path" {
  description = "Fully qualified project.dataset path for BigQuery DDL execution."
  value       = "${var.project_id}.${google_bigquery_dataset.staging_warehouse.dataset_id}"
}