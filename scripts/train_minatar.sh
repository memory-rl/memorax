for f in /home/mak72ze/memory-rl/scripts/train_*_minatar.sh; do
	bash "$f" &
done
wait
