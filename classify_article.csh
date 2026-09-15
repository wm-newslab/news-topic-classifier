#!/bin/tcsh
#SBATCH --job-name=localnews-article
#SBATCH --output=logs/article_%j.out
#SBATCH --error=logs/article_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load python
source .venv/bin/activate.csh

python -u localnews_classifier.py \
  --title "City opens new bus route" \
  --content "The new route connects downtown with nearby neighborhoods."
exit $status
