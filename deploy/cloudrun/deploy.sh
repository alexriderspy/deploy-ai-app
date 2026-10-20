#!/bin/bash
# Deploy the API to Google Cloud Run. Needs the gcloud CLI, logged in to a project with billing on.
# In the video this is explained step by step but NOT run: there are no cloud credentials on my laptop.
set -euo pipefail

PROJECT_ID=${PROJECT_ID:?set PROJECT_ID}
REGION=${REGION:-us-central1}
GITHUB_USER=${GITHUB_USER:-alexriderspy}
TAG=${TAG:-latest}
BUCKET=gs://$PROJECT_ID-models
SA=issue-triage@$PROJECT_ID.iam.gserviceaccount.com

# 1. switch on the services we use
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com --project "$PROJECT_ID"

# 2. our image is private on GHCR, so Artifact Registry pulls it for Cloud Run (a "remote repository").
#    GHCR_TOKEN is a GitHub token with read:packages scope.
printf %s "$GHCR_TOKEN" | gcloud secrets create ghcr-token --data-file=- --project "$PROJECT_ID"
gcloud artifacts repositories create ghcr --project "$PROJECT_ID" --location "$REGION" \
  --repository-format=docker --mode=remote-repository \
  --remote-repo-config-desc="GitHub Container Registry" --remote-docker-repo=https://ghcr.io \
  --remote-username="$GITHUB_USER" \
  --remote-password-secret-version="projects/$PROJECT_ID/secrets/ghcr-token/versions/latest"

# 3. the weights go to Cloud Storage, not into the image
gcloud storage buckets create "$BUCKET" --location "$REGION" --project "$PROJECT_ID"
gcloud storage cp -r model "$BUCKET/issue-triage"

# 4. the API key lives in Secret Manager, never in the code or the image
openssl rand -hex 24 | gcloud secrets create triage-api-keys --data-file=- --project "$PROJECT_ID"

# 5. a service account that can only read the bucket and the API key secret
gcloud iam service-accounts create issue-triage --project "$PROJECT_ID"
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member "serviceAccount:$SA" --role roles/storage.objectViewer
gcloud secrets add-iam-policy-binding triage-api-keys --project "$PROJECT_ID" \
  --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor

# 6. deploy: 4 CPUs, 8 GiB, up to 8 requests per instance, scale from 0 to 3 instances
gcloud run deploy issue-triage --project "$PROJECT_ID" --region "$REGION" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/ghcr/$GITHUB_USER/deploy-ai-app:$TAG" \
  --service-account "$SA" \
  --cpu 4 --memory 8Gi --concurrency 8 --min-instances 0 --max-instances 3 \
  --timeout 60 --cpu-boost --execution-environment gen2 \
  --add-volume name=models,type=cloud-storage,bucket="$PROJECT_ID-models",readonly=true \
  --add-volume-mount volume=models,mount-path=/models \
  --set-env-vars WEIGHTS_DIR=/models/issue-triage,MAX_BATCH_SIZE=8,OMP_NUM_THREADS=4 \
  --set-secrets API_KEYS=triage-api-keys:latest \
  --allow-unauthenticated

# 7. try it
URL=$(gcloud run services describe issue-triage --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')
KEY=$(gcloud secrets versions access latest --secret triage-api-keys --project "$PROJECT_ID")
curl -s "$URL/predict" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"title": "Engine crashes with CUDA OOM when loading Qwen3 with tensor parallel 4"}'
