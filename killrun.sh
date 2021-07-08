pids="$(ps aux | grep 'python run.py' | grep -v 'grep' | tr -s ' ' | cut -d ' ' -f 2)"
for pid in $pids
do
    if ps -p $pid > /dev/null
    then
        kill -9 $pid
    else
        continue
    fi
done
