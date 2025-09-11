for f in /home/mak72ze/memory-rl/scripts/train_*_memory_chain.sh; do
	bash "$f" &
done
wait
