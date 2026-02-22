#!/bin/bash
# Build multi-session overlay for all canonical-delivery images.
# Each base image gets a new tag with "-ms" suffix.
#
# Usage: bash build_all.sh [--dry-run]

set -e
cd "$(dirname "$0")"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

echo "=== Building multi-session overlay images ==="
echo "Context directory: $(pwd)"
echo ""

total=0
built=0
failed=0

for base_image in $(sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep '^canonical-delivery:' | sort); do
    total=$((total + 1))
    # e.g., canonical-delivery:task_xxx → canonical-ms:task_xxx
    tag_part="${base_image#canonical-delivery:}"
    ms_image="canonical-ms:${tag_part}"

    echo "[$total] ${base_image} → ${ms_image}"

    if $DRY_RUN; then
        echo "  [dry-run] Would build ${ms_image}"
        built=$((built + 1))
        continue
    fi

    if sudo docker build \
        --build-arg "BASE_IMAGE=${base_image}" \
        -t "${ms_image}" \
        -f Dockerfile \
        . 2>&1 | tail -3; then
        built=$((built + 1))
        echo "  OK"
    else
        failed=$((failed + 1))
        echo "  FAILED"
    fi
    echo ""
done

echo "=== Done ==="
echo "Total: $total | Built: $built | Failed: $failed"
