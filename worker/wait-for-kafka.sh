#!/bin/sh

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready at $KAFKA_BOOTSTRAP_SERVERS..."


  sleep 200


echo "Kafka is ready. Starting worker..."
exec python worker.py
