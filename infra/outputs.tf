output "alb_dns_name" {
  description = "ALB DNS name — point your MCP client here"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing images"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

output "github_deploy_role_arn" {
  description = "ARN of the role GitHub Actions assumes via OIDC to deploy"
  value       = aws_iam_role.github_deploy.arn
}

output "mcp_endpoint" {
  description = "MCP Streamable HTTP endpoint"
  value       = "http://${aws_lb.main.dns_name}/mcp"
}
