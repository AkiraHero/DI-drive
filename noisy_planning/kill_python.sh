ps -ef|grep python|awk '{print "kill -9 "$2 "&"}'|sh
