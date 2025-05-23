FROM bde2020/hadoop-base:2.0.0-hadoop3.2.1-java8

COPY upload-to-hdfs.sh /upload-to-hdfs.sh
RUN chmod +x /upload-to-hdfs.sh

CMD ["/upload-to-hdfs.sh"]