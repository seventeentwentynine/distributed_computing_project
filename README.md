# distributed_computing_project

CMPT 756, 2026 project repository.

## 0) Download the ONNX model first

Before building containers, download `model.onnx` and place it in the model version folder:

- requested path: `triton_workspace/model_repository/oberta_news/1/model.onnx`
- current repo manifest path: `triton_workspace/model_repository/roberta_news/1/model.onnx`

Model link: https://huggingface.co/echonode/cmpt756_ML_model/blob/main/onnx/model.onnx

Example download command:

```bash
mkdir -p triton_workspace/model_repository/roberta_news/1
curl -L "https://huggingface.co/echonode/cmpt756_ML_model/resolve/main/onnx/model.onnx" \
  -o triton_workspace/model_repository/roberta_news/1/model.onnx
```

## 1) Build containers and push to Google Artifact Registry

Set environment variables:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export REPO="ml-repo"
export TAG="v1"

export FRONTEND_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/fastapi-frontend:${TAG}"
export TRITON_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/triton-sidecar:${TAG}"
```

Authenticate and create Artifact Registry repo (skip create if already exists):

```bash
gcloud auth login
gcloud config set project "${PROJECT_ID}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Containers for roberta-news-classifier"
```

Build and push images:

```bash
docker build -t "${FRONTEND_IMAGE}" -f fastapi_app/Dockerfile fastapi_app
docker push "${FRONTEND_IMAGE}"

docker build -t "${TRITON_IMAGE}" -f triton_workspace/Dockerfile.triton triton_workspace
docker push "${TRITON_IMAGE}"
```

Update image references in both deployment manifests by replacing `{ACCOUNT}` with `${PROJECT_ID}`:

```bash
sed -i '' "s/{ACCOUNT}/${PROJECT_ID}/g" gke_deployment.yaml
sed -i '' "s/{ACCOUNT}/${PROJECT_ID}/g" service.yaml
```

## 2) Serve on Cloud Run (GPU sidecar deployment)

Cloud Run service manifest is in `service.yaml`.

Deploy:

```bash
gcloud run services replace service.yaml --region "${REGION}"
```

Get service URL:

```bash
gcloud run services describe roberta-news-classifier \
  --region "${REGION}" \
  --format='value(status.url)'
```

## 3) Serve on GKE

Kubernetes manifest is in `gke_deployment.yaml`.

If your cluster already exists, add the L4 GPU node pool:

```bash
gcloud container node-pools create l4-pool \
  --cluster YOUR_CLUSTER \
  --location us-central1 \
  --node-locations us-central1-b \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=default \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 3
```

Connect `kubectl` to cluster and deploy:

```bash
gcloud container clusters get-credentials YOUR_CLUSTER --location us-central1
kubectl apply -f gke_deployment.yaml
```

Check rollout and external ingress IP:

```bash
kubectl get pods
kubectl get svc roberta-news-classifier
kubectl get ingress roberta-news-classifier
```
