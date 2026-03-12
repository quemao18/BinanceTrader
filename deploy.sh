#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-binance-trader-job}"
SCHEDULER_JOB_NAME="${SCHEDULER_JOB_NAME:-run-bot-job}"
IMAGE_NAME="${IMAGE_NAME:-binance-trader}"
SCHEDULE="${SCHEDULE:-*/15 * * * *}"
TRADING_PAIR="${TRADING_PAIR:-BTCUSDT}"
TIMEFRAME="${TIMEFRAME:-15m}"
TESTNET="${TESTNET:-false}"
TRADE_MODE="${TRADE_MODE:-spot}"
SINGLE_RUN="${SINGLE_RUN:-true}"
MAX_CYCLES_PER_RUN="${MAX_CYCLES_PER_RUN:-1}"
SLEEP_BETWEEN_CYCLES_SEC="${SLEEP_BETWEEN_CYCLES_SEC:-30}"
ONLY_NEW_CANDLE="${ONLY_NEW_CANDLE:-true}"
SKIP_FETCH_CURRENCIES="${SKIP_FETCH_CURRENCIES:-true}"
MAX_POSITION_PCT="${MAX_POSITION_PCT:-0.25}"
BUY_THRESHOLD="${BUY_THRESHOLD:-0.60}"
SELL_THRESHOLD="${SELL_THRESHOLD:-0.40}"
AUTO_LOAD_BEST_THRESHOLDS="${AUTO_LOAD_BEST_THRESHOLDS:-true}"
BEST_THRESHOLDS_PATH="${BEST_THRESHOLDS_PATH:-best_thresholds.json}"
STATE_FILE="${STATE_FILE:-bot_state.json}"
KILL_SWITCH_FILE="${KILL_SWITCH_FILE:-.pause_trading}"
REQUIRE_GO_NO_GO="${REQUIRE_GO_NO_GO:-true}"
TRAINING_METRICS_PATH="${TRAINING_METRICS_PATH:-training_metrics.json}"
RUN_NOW="${RUN_NOW:-false}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd gcloud

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required."
  echo "Example: PROJECT_ID=binance-trader-20260309 ./deploy.sh"
  exit 1
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

API_KEY="${API_KEY:-}"
SECRET="${SECRET:-}"

if [[ -z "$API_KEY" || -z "$SECRET" ]]; then
  echo "API_KEY and SECRET are required in environment or .env"
  exit 1
fi

IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"

echo "Using project: ${PROJECT_ID}"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "Enabling required APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com >/dev/null

echo "Building and pushing image: ${IMAGE_URI}"
gcloud builds submit --tag "$IMAGE_URI"

ensure_secret() {
  local secret_name="$1"
  local secret_value="$2"
  if gcloud secrets describe "$secret_name" >/dev/null 2>&1; then
    printf '%s' "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=- >/dev/null
  else
    printf '%s' "$secret_value" | gcloud secrets create "$secret_name" --replication-policy=automatic --data-file=- >/dev/null
  fi
}

echo "Upserting secrets..."
ensure_secret "binance-api-key" "$API_KEY"
ensure_secret "binance-secret" "$SECRET"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
RUN_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_NUMBER}/jobs/${JOB_NAME}:run"

for s in binance-api-key binance-secret; do
  gcloud secrets add-iam-policy-binding "$s" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor" >/dev/null
 done

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/run.invoker" >/dev/null

echo "Creating/updating Cloud Run Job..."
if gcloud run jobs describe "$JOB_NAME" --region "$REGION" >/dev/null 2>&1; then
  gcloud run jobs update "$JOB_NAME" \
    --region "$REGION" \
    --image "$IMAGE_URI" \
    --set-secrets API_KEY=binance-api-key:latest,SECRET=binance-secret:latest \
    --set-env-vars TRADING_PAIR="$TRADING_PAIR",TIMEFRAME="$TIMEFRAME",TESTNET="$TESTNET",TRADE_MODE="$TRADE_MODE",SINGLE_RUN="$SINGLE_RUN",MAX_CYCLES_PER_RUN="$MAX_CYCLES_PER_RUN",SLEEP_BETWEEN_CYCLES_SEC="$SLEEP_BETWEEN_CYCLES_SEC",ONLY_NEW_CANDLE="$ONLY_NEW_CANDLE",SKIP_FETCH_CURRENCIES="$SKIP_FETCH_CURRENCIES",MAX_POSITION_PCT="$MAX_POSITION_PCT",BUY_THRESHOLD="$BUY_THRESHOLD",SELL_THRESHOLD="$SELL_THRESHOLD",AUTO_LOAD_BEST_THRESHOLDS="$AUTO_LOAD_BEST_THRESHOLDS",BEST_THRESHOLDS_PATH="$BEST_THRESHOLDS_PATH",STATE_FILE="$STATE_FILE",KILL_SWITCH_FILE="$KILL_SWITCH_FILE",REQUIRE_GO_NO_GO="$REQUIRE_GO_NO_GO",TRAINING_METRICS_PATH="$TRAINING_METRICS_PATH" \
    --max-retries=0 \
    --task-timeout=3600s >/dev/null
else
  gcloud run jobs create "$JOB_NAME" \
    --region "$REGION" \
    --image "$IMAGE_URI" \
    --set-secrets API_KEY=binance-api-key:latest,SECRET=binance-secret:latest \
    --set-env-vars TRADING_PAIR="$TRADING_PAIR",TIMEFRAME="$TIMEFRAME",TESTNET="$TESTNET",TRADE_MODE="$TRADE_MODE",SINGLE_RUN="$SINGLE_RUN",MAX_CYCLES_PER_RUN="$MAX_CYCLES_PER_RUN",SLEEP_BETWEEN_CYCLES_SEC="$SLEEP_BETWEEN_CYCLES_SEC",ONLY_NEW_CANDLE="$ONLY_NEW_CANDLE",SKIP_FETCH_CURRENCIES="$SKIP_FETCH_CURRENCIES",MAX_POSITION_PCT="$MAX_POSITION_PCT",BUY_THRESHOLD="$BUY_THRESHOLD",SELL_THRESHOLD="$SELL_THRESHOLD",AUTO_LOAD_BEST_THRESHOLDS="$AUTO_LOAD_BEST_THRESHOLDS",BEST_THRESHOLDS_PATH="$BEST_THRESHOLDS_PATH",STATE_FILE="$STATE_FILE",KILL_SWITCH_FILE="$KILL_SWITCH_FILE",REQUIRE_GO_NO_GO="$REQUIRE_GO_NO_GO",TRAINING_METRICS_PATH="$TRAINING_METRICS_PATH" \
    --max-retries=0 \
    --task-timeout=3600s >/dev/null
fi

echo "Creating/updating Cloud Scheduler..."
if gcloud scheduler jobs describe "$SCHEDULER_JOB_NAME" --location "$REGION" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "$SCHEDULER_JOB_NAME" \
    --location "$REGION" \
    --schedule "$SCHEDULE" \
    --uri "$RUN_URI" \
    --http-method POST \
    --oauth-service-account-email "$RUNTIME_SA" \
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" \
    --update-headers "Content-Type=application/json" \
    --message-body '{}' >/dev/null
else
  gcloud scheduler jobs create http "$SCHEDULER_JOB_NAME" \
    --location "$REGION" \
    --schedule "$SCHEDULE" \
    --uri "$RUN_URI" \
    --http-method POST \
    --oauth-service-account-email "$RUNTIME_SA" \
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" \
    --headers "Content-Type=application/json" \
    --message-body '{}' >/dev/null
fi

echo "Deployment complete."
echo "Image: ${IMAGE_URI}"
echo "Job: ${JOB_NAME}"
echo "Scheduler: ${SCHEDULER_JOB_NAME} (${SCHEDULE})"

if [[ "$RUN_NOW" == "true" ]]; then
  EXEC_NAME="$(gcloud run jobs execute "$JOB_NAME" --region "$REGION" --format='value(metadata.name)')"
  echo "Started execution: ${EXEC_NAME}"
fi
