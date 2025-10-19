# Path: tests/smoke_test.ps1
# Run in PowerShell (Developer PowerShell or standard PowerShell with venv activated)

$ErrorActionPreference = "Stop"

Write-Host "1) Running load_to_neo4j.py..."
python load_to_neo4j.py
if ($LASTEXITCODE -ne 0) { Write-Host "neo4j failed"; exit 1 }

Write-Host "2) Running pinecone_upload.py..."
python pinecone_upload.py
if ($LASTEXITCODE -ne 0) { Write-Host "pinecone upload failed"; exit 2 }

Write-Host "3) Running visualize_graph.py..."
python visualize_graph.py
if ($LASTEXITCODE -ne 0) { Write-Host "viz failed"; exit 3 }

Write-Host "4) Import test for hybrid_chat..."
python -c "import hybrid_chat; print('hybrid import OK')"
if ($LASTEXITCODE -ne 0) { Write-Host "hybrid import failed"; exit 4 }

Write-Host "SMOKE TESTS PASSED"
exit 0
