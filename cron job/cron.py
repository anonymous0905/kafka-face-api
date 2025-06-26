import time
import requests
from supabase import create_client, Client
import schedule

# --- CONFIGURATION ---
SUPABASE_URL = "https://agrdrscvweokrrcsabsz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFncmRyc2N2d2Vva3JyY3NhYnN6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTY5MzU2NCwiZXhwIjoyMDY1MjY5NTY0fQ.OaquJNKMROqZb5qRJufyZhAGgGY6UlQbXcawpZEsP7E"
TARGET_URL = "http://localhost:8001/enqueue"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def check_and_enqueue():
    print("Checking flagged records with null reason...")

    response = supabase.table("attendance_logs").select("id") \
        .eq("verified", False) \
        .eq("flagged", True) \
        .is_("reason_flagged", None) \
        .execute()

    records = response.data
    print(f"Found {len(records)} records to process.")

    for record in records:
        payload = {"id": record["id"]}
        try:
            res = requests.post(TARGET_URL, json=payload)
            res.raise_for_status()
            print(f"✅ Sent ID {record['id']} successfully")
        except requests.RequestException as e:
            print(f"❌ Failed to send ID {record['id']}: {e}")

# --- Run once on startup ---
check_and_enqueue()
# --- SCHEDULING ---
schedule.every(2).hours.do(check_and_enqueue)

print("⏰ Job scheduled to run every 2 hours.")
while True:
    schedule.run_pending()
    time.sleep(60)
