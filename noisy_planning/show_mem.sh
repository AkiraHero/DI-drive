ps -U $USER --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB"}'
