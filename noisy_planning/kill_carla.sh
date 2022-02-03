ps -ef|grep Carla|awk '{print "kill -9 "$2 "&"}'|sh

