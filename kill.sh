for pid in $(ps | grep "Chrome" | awk '{print $1}'); do kill -9 $pid; done
for pid in $(ps | grep "chrome" | awk '{print $1}'); do kill -9 $pid; done
