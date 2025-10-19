# Path: tests/smoke_test.sh
#!/usr/bin/env bash
set -euo pipefail

# Run Neo4j loader
echo "1) Running load_to_neo4j.py..."
python load_to_neo4j.py || { echo "neo4j failed"; exit 1; }

# Run Pinecone uploader
echo "2) Running pinecone_upload.py..."
python pinecone_upload.py || { echo "pinecone upload failed"; exit 2; }

# Run graph visualizer
echo "3) Running visualize_graph.py..."
python visualize_graph.py || { echo "viz failed"; exit 3; }

# Ensure hybrid_chat imports (interactive — we only test importability)
echo "4) Import test for hybrid_chat..."
python -c "import hybrid_chat; print('hybrid import OK')" || { echo "hybrid import failed"; exit 4; }

echo "SMOKE TESTS PASSED"
exit 0
