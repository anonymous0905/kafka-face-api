import os, asyncio, json, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
from aiokafka import AIOKafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "jobs"

app = FastAPI(title="Job-Producer")
producer: AIOKafkaProducer | None = None
log = logging.getLogger("producer")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

class Job(BaseModel):
    id: constr(strip_whitespace=True, min_length=1)

async def connect_producer_with_retry():
    global producer
    while True:
        producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
        try:
            await producer.start()
            log.info("✓ Producer connected to Kafka → %s", KAFKA_BOOTSTRAP)
            break
        except Exception as e:
            log.warning("Kafka not ready (%s). Retrying in 2s...", str(e))
            await asyncio.sleep(2)

@app.on_event("startup")
async def startup():
    await connect_producer_with_retry()

@app.on_event("shutdown")
async def shutdown():
    if producer:
        await producer.stop()
        log.info("Producer stopped")

@app.post("/enqueue", status_code=202)
async def enqueue(job: Job):
    try:
        payload = json.dumps(job.model_dump()).encode()
        await producer.send_and_wait(TOPIC, payload)
        return {"status": "queued", **job.model_dump()}
    except Exception as e:
        log.exception("Kafka send failed")
        raise HTTPException(500, "Could not enqueue") from e
