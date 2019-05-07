sudo yum install tmux
sudo yum install htop
sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
#sudo pip-3.4 install matplotlib
hadoop fs -put *.dat /user/hadoop/
sudo pip-3.4 install findspark

spark-submit spark_v2.py --num-executors 4
