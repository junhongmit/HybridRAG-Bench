#!/bin/bash

# # backup the progress file
# backup_progress_file() {
#     local file="$1" # The file to backup
#     if [ -z "$file" ]; then
#         echo "Usage: $0 <filename>"
#         exit 1
#     fi

#     if [ ! -f "$file" ]; then
#         echo "Error: File '$file' not found."
#         exit 1
#     fi

#     mv "$file" "${file}.$(date +%Y%m%d_%H%M%S)"
#     echo "Backup created: ${file}.$(date +%Y%m%d_%H%M%S)"
# }
# backup_progress_file "results/update_movie_kg_progress.json"

# reset the database
current_dir=$(pwd)
neo4j_dir="$HOME/neo4j"
cd $neo4j_dir

bin/neo4j stop
rm -r data/databases

cp -r data/backup/databases data/databases
echo "replaced the databases folder"

rm -r data/transactions
cp -r data/backup/transactions data/transactions
echo "replaced the transactions folder"

bin/neo4j start

cd $current_dir
