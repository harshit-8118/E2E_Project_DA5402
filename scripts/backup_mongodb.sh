#!/bin/bash
# Back up MongoDB data without wiping the volume
# bash scripts/restore_mongodb.sh ./backups/dermai_20260418_120000
set -e
mkdir -p ./backups

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/dermai_${TIMESTAMP}"

echo "Starting MongoDB backup → ${BACKUP_DIR}"

docker exec dermai-mongodb mongodump \
  --uri="mongodb://localhost:27017" \
  --db="${MONGO_DB:-skin_disease_detection}" \
  --out="/tmp/dump_${TIMESTAMP}" \
  --quiet

docker cp "dermai-mongodb:/tmp/dump_${TIMESTAMP}" "${BACKUP_DIR}"

echo "✅ Backup saved to ${BACKUP_DIR}"
echo "   Restore with: bash scripts/restore_mongodb.sh ${BACKUP_DIR}"