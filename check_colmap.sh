#!/bin/bash
# État de l'installation COLMAP et de la préparation du solve.
if which colmap >/dev/null 2>&1; then
  echo "✓ COLMAP prêt : $(colmap --help 2>&1 | head -1)"
  echo "  lancer : python -m hotel_pipeline.cli reconstruction run welcominns-boucherville --backend colmap_incremental"
else
  echo "⏳ COLMAP en construction"
  echo "   paquet en cours : $(ps aux | grep '[b]uild.rb' | awk '{print $NF}' | sed 's|.*/||' | head -1)"
  python3 -c "
import subprocess
need=set(subprocess.run(['brew','deps','colmap'],capture_output=True,text=True).stdout.split())
have=set(subprocess.run(['brew','list','--formula'],capture_output=True,text=True).stdout.split())
print(f'   dépendances restantes : {len(need-have)}/{len(need)}')
"
  pgrep -f "brew.rb install colmap" >/dev/null && echo "   processus brew : actif" || echo "   ⚠ processus brew : ARRÊTÉ"
fi
echo "   corpus prêt : $(ls work/welcominns-boucherville/05_colmap/front_solve/images/*.jpg 2>/dev/null | wc -l | tr -d ' ') images"
