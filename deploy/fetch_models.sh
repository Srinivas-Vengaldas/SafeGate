#!/usr/bin/env bash
# Downloads the newest trained models from GitHub Actions artifacts into OUT_DIR:
#   OUT_DIR/injection-onnx, OUT_DIR/injection-threshold  (Train injection classifier)
#   OUT_DIR/toxicity-onnx                                (Export toxicity model)
# Needs GH_TOKEN with actions:read. Usage: deploy/fetch_models.sh OUT_DIR
set -euo pipefail
out=$1
mkdir -p "$out"
for run_id in $(gh run list --workflow train.yml --status success --limit 20 \
    --json databaseId --jq '.[].databaseId'); do
  if gh run download "$run_id" --name injection-onnx --dir "$out/injection-onnx" 2>/dev/null; then
    gh run download "$run_id" --name reports --dir "$out/reports"
    python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['threshold'])" \
      "$out/reports/onnx/metrics.json" > "$out/injection-threshold"
    if [ ! -f "$out/injection-onnx/model.onnx.data" ]; then
      pip install --quiet onnx numpy
      python -m training.export_onnx --externalize "$out/injection-onnx"
    fi
    echo "injection classifier: training run $run_id, threshold $(cat "$out/injection-threshold")"
    break
  fi
done
run_id=$(gh run list --workflow toxicity.yml --status success --limit 1 \
  --json databaseId --jq '.[0].databaseId // empty')
if [ -n "$run_id" ]; then
  gh run download "$run_id" --name toxicity-onnx --dir "$out/toxicity-onnx"
  echo "toxicity model: export run $run_id ($(cat "$out/toxicity-onnx/SOURCE"))"
fi
