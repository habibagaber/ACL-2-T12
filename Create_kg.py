import csv
import os
import sys
from neo4j import GraphDatabase, basic_auth

# -------------------------
# Utility helpers

# -------------------------
def load_config(path="config.txt"):
    cfg = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg

def safe_float(val, default=0.0):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default

def safe_int(val, default=None):
    try:
        if val is None or val == "":
            return default
        return int(val)
    except Exception:
        return default

# -------------------------
# Neo4j wrapper
# -------------------------
def load_config(path="config.txt"):
    cfg = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg

# -------------------------
# Neo4j wrapper
# -------------------------
class Neo4jRunner:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    def run(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            # Fetch all records into a list before returning
            return list(result)

    def close(self):
        self.driver.close()

# -------------------------
# Clear DB
# -------------------------
def clear_database(runner):
    print("Clearing existing graph (MATCH (n) DETACH DELETE n)...")
    runner.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")

# -------------------------
# Load Hotels (batch)
# -------------------------
def load_hotels(runner, path="hotels.csv", batch_size=500):
    if not os.path.exists(path):
        print(f"[load_hotels] WARNING: {path} not found. Skipping hotels load.")
        return
    print(f"Loading hotels from {path} ...")
    batch = []
    count = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append({
                "hotel_id": row.get("hotel_id") or row.get("id") or row.get("hotelId"),
                "hotel_name": row.get("hotel_name") or row.get("name"),
                "city": row.get("city"),
                "country": row.get("country"),
                "lat": safe_float(row.get("lat")),
                "lon": safe_float(row.get("lon")),
                "star_rating": safe_float(row.get("star_rating")),
                "cleanliness_base": safe_float(row.get("cleanliness_base")),
                "comfort_base": safe_float(row.get("comfort_base")),
                "facilities_base": safe_float(row.get("facilities_base"))
            })
            if len(batch) >= batch_size:
                runner.run("""
                UNWIND $batch AS h
                MERGE (hotel:Hotel {hotel_id: h.hotel_id})
                SET hotel.name = h.hotel_name,
                    hotel.city = h.city,
                    hotel.country = h.country,
                    hotel.lat = h.lat,
                    hotel.lon = h.lon,
                    hotel.star_rating = h.star_rating,
                    hotel.cleanliness_base = h.cleanliness_base,
                    hotel.comfort_base = h.comfort_base,
                    hotel.facilities_base = h.facilities_base
                """, {"batch": batch})
                count += len(batch)
                batch = []
    if batch:
        runner.run("""
        UNWIND $batch AS h
        MERGE (hotel:Hotel {hotel_id: h.hotel_id})
        SET hotel.name = h.hotel_name,
            hotel.city = h.city,
            hotel.country = h.country,
            hotel.lat = h.lat,
            hotel.lon = h.lon,
            hotel.star_rating = h.star_rating,
            hotel.cleanliness_base = h.cleanliness_base,
            hotel.comfort_base = h.comfort_base,
            hotel.facilities_base = h.facilities_base
        """, {"batch": batch})
        count += len(batch)
    print(f"Hotels loaded: {count}")

# -------------------------
# Load Users (Travellers) (batch)
# -------------------------
def load_users(runner, path="users.csv", batch_size=500):
    if not os.path.exists(path):
        print(f"[load_users] WARNING: {path} not found. Skipping users load.")
        return

    # Map age_group to numeric age (midpoint)
    AGE_GROUP_TO_NUM = {
        "18-24": 21,
        "25-34": 29,
        "35-44": 39,
        "45-54": 49,
        "55+": 60
    }

    print(f"Loading travellers from {path} ...")
    batch = []
    count = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age_group = row.get("age_group")
            age = AGE_GROUP_TO_NUM.get(age_group)
            batch.append({
                "user_id": row.get("user_id"),
                "age": age,
                "gender": row.get("user_gender"),
                "type": row.get("traveller_type") or row.get("type"),
                "country": row.get("country")
                
            })
            if len(batch) >= batch_size:
                runner.run("""
                UNWIND $batch AS t
                MERGE (traveller:Traveller {user_id: t.user_id})
                SET traveller.age = t.age,
                    traveller.gender = t.gender,
                    traveller.type = t.type,
                    traveller.country = t.country
                """, {"batch": batch})
                count += len(batch)
                batch = []

    if batch:
        runner.run("""
        UNWIND $batch AS t
        MERGE (traveller:Traveller {user_id: t.user_id})
        SET traveller.age = t.age,
            traveller.gender = t.gender,
            traveller.type = t.type
        """, {"batch": batch})
        count += len(batch)

    print(f"Travellers loaded: {count}")

# -------------------------
# Load Reviews (batch)
# -------------------------
def load_reviews(runner, path="reviews.csv", batch_size=5000):
    if not os.path.exists(path):
        print(f"[load_reviews] WARNING: {path} not found. Skipping reviews load.")
        return
    print(f"Loading reviews from {path} ...")
    batch = []
    count = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append({
                "review_id": row.get("review_id") or row.get("id"),
                "text": row.get("text") or row.get("review_text") or "",
                "date": row.get("date") or "",
                "user_id": row.get("user_id"),
                "hotel_id": row.get("hotel_id"),
                "score_overall": safe_float(row.get("score_overall")),
                "score_cleanliness": safe_float(row.get("score_cleanliness")),
                "score_comfort": safe_float(row.get("score_comfort")),
                "score_facilities": safe_float(row.get("score_facilities")),
                "score_location": safe_float(row.get("score_location")),
                "score_staff": safe_float(row.get("score_staff")),
                "score_value_for_money": safe_float(row.get("score_value_for_money"))
            })
            if len(batch) >= batch_size:
                runner.run("""
                UNWIND $batch AS r
                MERGE (review:Review {review_id: r.review_id})
                SET review.text = r.text,
                    review.date = r.date,
                    review.user_id = r.user_id,
                    review.hotel_id = r.hotel_id,
                    review.score_overall = r.score_overall,
                    review.score_cleanliness = r.score_cleanliness,
                    review.score_comfort = r.score_comfort,
                    review.score_facilities = r.score_facilities,
                    review.score_location = r.score_location,
                    review.score_staff = r.score_staff,
                    review.score_value_for_money = r.score_value_for_money
                """, {"batch": batch})
                count += len(batch)
                batch = []
    if batch:
        runner.run("""
        UNWIND $batch AS r
        MERGE (review:Review {review_id: r.review_id})
        SET review.text = r.text,
            review.date = r.date,
            review.user_id = r.user_id,
            review.hotel_id = r.hotel_id,
            review.score_overall = r.score_overall,
            review.score_cleanliness = r.score_cleanliness,
            review.score_comfort = r.score_comfort,
            review.score_facilities = r.score_facilities,
            review.score_location = r.score_location,
            review.score_staff = r.score_staff,
            review.score_value_for_money = r.score_value_for_money
        """, {"batch": batch})
        count += len(batch)
    print(f"Reviews loaded: {count}")

# -------------------------
# Cities & Countries
# -------------------------
def load_cities_and_countries_from_hotels(runner):
    print("Creating City and Country nodes from hotels...")
    runner.run("""
    MATCH (h:Hotel)
    WHERE h.city IS NOT NULL
    MERGE (c:City {name: h.city})
    """)
    runner.run("""
    MATCH (h:Hotel)
    WHERE h.country IS NOT NULL
    MERGE (co:Country {name: h.country})
    """)
    print("City and Country nodes created from hotels.")

# -------------------------
# Visa data
# -------------------------
def load_visa(runner, path="visa.csv"):
    if not os.path.exists(path):
        print(f"[load_visa] WARNING: {path} not found. Skipping visa load.")
        return
    print(f"Loading visa data from {path} ...")
    batch = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append({
                "from": row.get("from") or row.get("country_from"),
                "to": row.get("to") or row.get("country_to"),
                "requires": row.get("requires") or row.get("requires_visa"),
                "visa_type": row.get("visa_type") or row.get("type") or ""
            })
            if len(batch) >= 500:
                runner.run("""
                UNWIND $batch AS v
                MERGE (c1:Country {name: v.from})
                MERGE (c2:Country {name: v.to})
                MERGE (c1)-[r:NEEDS_VISA]->(c2)
                SET r.visa_type = v.visa_type, r.requires = v.requires
                """, {"batch": batch})
                batch = []
    if batch:
        runner.run("""
        UNWIND $batch AS v
        MERGE (c1:Country {name: v.from})
        MERGE (c2:Country {name: v.to})
        MERGE (c1)-[r:NEEDS_VISA]->(c2)
        SET r.visa_type = v.visa_type, r.requires = v.requires
        """, {"batch": batch})
    print("Visa data loaded.")

# -------------------------
# Relationships
# -------------------------
def create_relationships(runner):
    print("Creating FROM_COUNTRY relationships (Traveller -> Country)...")
    runner.run("""
    MATCH (t:Traveller)
    WHERE t.country IS NOT NULL
    MERGE (co:Country {name: t.country})
    MERGE (t)-[:FROM_COUNTRY]->(co)
    """)
    print("Creating Hotel -> City relationships (LOCATED_IN)...")
    runner.run("""
    MATCH (h:Hotel)
    WHERE h.city IS NOT NULL
    MATCH (c:City {name: h.city})
    MERGE (h)-[:LOCATED_IN]->(c)
    """)
    print("Creating City -> Country relationships (LOCATED_IN)...")
    runner.run("""
    MATCH (h:Hotel)
    WHERE h.country IS NOT NULL AND h.city IS NOT NULL
    MATCH (c:City {name: h.city}), (co:Country {name: h.country})
    MERGE (c)-[:LOCATED_IN]->(co)
    """)
    print("Creating Review -> Hotel (REVIEWED) and Traveller -> Hotel (STAYED_AT) relationships...")
    runner.run("""
    MATCH (r:Review), (h:Hotel)
    WHERE r.hotel_id = h.hotel_id
    MERGE (r)-[:REVIEWED]->(h)
    WITH r, h
    MATCH (t:Traveller)
    WHERE t.user_id = r.user_id
    MERGE (t)-[:STAYED_AT]->(h)
    """)
    print("Creating Traveller -> Review (WROTE) relationships...")
    runner.run("""
    MATCH (t:Traveller), (r:Review)
    WHERE t.user_id = r.user_id
    MERGE (t)-[:WROTE]->(r)
    """)
    print("All relationships created.")

# -------------------------
# Average review score
# -------------------------
def calculate_average_review_score(runner, batch_size=500):
    print("Calculating average_reviews_score per Hotel (batch)...")

    # Fetch all hotel IDs upfront into a list
    records = runner.run("MATCH (h:Hotel) RETURN h.hotel_id AS hid")
    hotel_ids = [record["hid"] for record in records]
    
    total = len(hotel_ids)
    print(f"Total hotels: {total}")

    batch = []
    for idx, hid in enumerate(hotel_ids, 1):
        batch.append({"hotel_id": hid})
        if len(batch) >= batch_size:
            runner.run("""
                UNWIND $batch AS row
                MATCH (h:Hotel {hotel_id: row.hotel_id})<-[:REVIEWED]-(r:Review)
                WITH h, AVG(r.score_overall) AS avg_score
                SET h.average_reviews_score = avg_score
            """, {"batch": batch})
            print(f"Processed batch {idx-len(batch)+1} to {idx}")
            batch = []

    # Process remaining hotels
    if batch:
        runner.run("""
            UNWIND $batch AS row
            MATCH (h:Hotel {hotel_id: row.hotel_id})<-[:REVIEWED]-(r:Review)
            WITH h, AVG(r.score_overall) AS avg_score
            SET h.average_reviews_score = avg_score
        """, {"batch": batch})
        print(f"Processed final batch of {len(batch)} hotels")

    print("Average review scores set.")

# -------------------------
# Exceeds Expectations
# -------------------------
# -------------------------
# Exceeds Expectations (fixed)
# -------------------------
def calculate_exceeds_expectations(runner):
    AGE_GROUPS = {
        "18-24": (18, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55+": (55, 200)
    }
    results = []
    
    for age_group, (age_min, age_max) in AGE_GROUPS.items():
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
        WHERE t.gender = 'Female' AND t.type = 'Solo'
          AND t.age >= $age_min AND t.age <= $age_max
        WITH h, AVG(r.score_cleanliness + r.score_comfort + r.score_facilities) AS review_avg,
             (h.cleanliness_base + h.comfort_base + h.facilities_base) AS base_total
        WHERE base_total >= review_avg
        RETURN
            MIN( ((base_total - review_avg)/review_avg)*100 ) AS min_improve,
            MAX( ((base_total - review_avg)/review_avg)*100 ) AS max_improve,
            AVG( ((base_total - review_avg)/review_avg)*100 ) AS avg_improve
        """
        records = runner.run(query, {"age_min": age_min, "age_max": age_max})
        row = records[0] if records else None
        if row:
            results.append({
                "age_group": age_group,
                "min_improve": round(row["min_improve"] or 0, 2),
                "max_improve": round(row["max_improve"] or 0, 2),
                "avg_improve": round(row["avg_improve"] or 0, 2)
            })
        else:
            results.append({
                "age_group": age_group,
                "min_improve": 0,
                "max_improve": 0,
                "avg_improve": 0
            })
    return results

# -------------------------
# Main
# -------------------------
def main():
    try:
        cfg = load_config("config.txt")
        print("Config loaded:", cfg)
    except Exception as e:
        print("ERROR loading config.txt:", e)
        sys.exit(1)

    uri = cfg.get("URI", "neo4j://localhost:7687")
    user = cfg.get("USERNAME", "neo4j")
    pwd = cfg.get("PASSWORD", "neo4j")

    print(f"Connecting to Neo4j at {uri} as {user} ...")
    runner = Neo4jRunner(uri, user, pwd)

    try:
        clear_database(runner)
        load_hotels(runner, "hotels.csv")
        load_users(runner, "users.csv")
        load_reviews(runner, "reviews.csv")
        load_cities_and_countries_from_hotels(runner)
        load_visa(runner, "visa.csv")
        create_relationships(runner)
        calculate_average_review_score(runner)

        print("\nCalculating Exceeds Expectations for Solo Female travellers...")
        exceed_stats = calculate_exceeds_expectations(runner)
        print("age_group | min_improve | max_improve | avg_improve")
        for row in exceed_stats:
            print(f"{row['age_group']:8} | {row['min_improve']:11} | {row['max_improve']:11} | {row['avg_improve']:11}")

        print("\nKnowledge Graph creation complete.")

    except Exception as e:
        print("ERROR during KG creation:", e)
    finally:
        runner.close()

if __name__ == "__main__":
    main()