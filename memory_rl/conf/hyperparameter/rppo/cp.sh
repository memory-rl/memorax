SRC=popgym_battleship_easy.yaml; DIR=../../environment
for f in "$DIR"/popgym_*.yaml; do
	  [ "$f" = "$DIR/popgym_battleship_easy.yaml" ] && continue
	    cp -f "$SRC" "$f"
    done

