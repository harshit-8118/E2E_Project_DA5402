#!/bin/bash
# Restore MongoDB from a backup WITHOUT wiping current data
# Usage: bash scripts/restore_mongodb.sh ./backups/dermai_20260418_120000

set -e
BACKUP_PATH="${1}"

if [ -z "${BACKUP_PATH}" ]; then
  echo "Usage: bash scripts/restore_mongodb.sh <backup_dir>"
  echo "Available backups:"
  ls ./backups/ 2>/dev/null || echo "  No backups found"
  exit 1
fi

echo "Restoring from ${BACKUP_PATH}"

# copy backup into container
docker cp "${BACKUP_PATH}" dermai-mongodb:/tmp/restore_data

# restore — --drop drops collection before restore, use --noIndexRestore if indexes exist
docker exec dermai-mongodb mongorestore \
  --uri="mongodb://localhost:27017" \
  --db="${MONGO_DB:-skin_disease_detection}" \
  --dir="/tmp/restore_data/${MONGO_DB:-skin_disease_detection}" \
  --drop \
  --quiet

echo "✅ Restore complete from ${BACKUP_PATH}"