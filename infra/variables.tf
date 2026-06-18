variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "atlas-counsel"
}

variable "container_image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "qdrant_url" {
  description = "Qdrant Cloud cluster URL (provisioned manually)"
  type        = string
  sensitive   = true
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

variable "container_cpu" {
  description = "CPU units per task (256 = 0.25 vCPU)"
  type        = number
  default     = 256
}

variable "container_memory" {
  description = "Memory per task in MiB"
  type        = number
  default     = 512
}
