# Deploying the public demo

The demo page (`/`), the API and the trained injection classifier run as one container on
Google Cloud Run. Cloud Run scales to zero when idle, so a portfolio demo stays inside the
always-free tier (2 million requests and 360,000 GB-seconds a month). The first visit after an
idle period takes a few seconds to start.

## One-time setup

1. Sign in at <https://console.cloud.google.com>, create a project (for example `safegate-demo`)
   and attach a billing account. A card is required, but this setup stays within the free tier.
   Optionally add a budget alert under **Billing → Budgets & alerts** (for example $1).
2. Open **Cloud Shell** (the terminal icon at the top right) and run, replacing the project ID:

   ```bash
   PROJECT=safegate-demo
   gcloud config set project $PROJECT
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   gcloud iam service-accounts create safegate-deployer
   SA=safegate-deployer@$PROJECT.iam.gserviceaccount.com
   for role in run.admin cloudbuild.builds.editor artifactregistry.admin storage.admin \
               iam.serviceAccountUser serviceusage.serviceUsageConsumer; do
     gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA --role roles/$role
   done
   gcloud iam service-accounts keys create key.json --iam-account $SA
   cat key.json
   ```

3. In this GitHub repo: **Settings → Secrets and variables → Actions**
   - Secrets tab: add `GCP_SA_KEY` with the full contents of `key.json`. Then delete the file
     in Cloud Shell (`rm key.json`).
   - Variables tab: add `GCP_PROJECT` with your project ID. Optionally add `GCP_REGION`
     (default `us-east1`).
4. Run **Deploy demo** from the Actions tab. The job summary prints the public URL.

Every later push to `main`, and every successful training run, redeploys automatically.

## Notes

- The image bundles the model from the latest successful **Train injection classifier** run
  and enables the injection rail with the threshold that run chose on validation data.
- The decision log uses SQLite inside the container, so it resets when the instance restarts.
  Production deployments use PostgreSQL via `docker-compose.yml`.
- No LLM key is configured, so the demo only screens prompts. Callers of
  `/v1/chat/completions` must send their own key, which is passed through and never stored.
