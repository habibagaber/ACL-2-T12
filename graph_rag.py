import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ============================================================================
# ENHANCED ENTITY EXTRACTOR WITH CSV SUPPORT
# ============================================================================

class EnhancedEntityExtractor:
    """
    Enhanced entity extractor that uses CSV data for better matching.
    """
    
    def __init__(self, csv_path: str = "hotels.csv"):
        """Initialize with hotel CSV data."""
        try:
            self.hotels_df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(self.hotels_df)} hotels from CSV")
            
            # Create lookup dictionaries for faster matching
            self.hotel_names = self.hotels_df['hotel_name'].str.lower().tolist()
            self.cities = self.hotels_df['city'].str.lower().unique().tolist()
            self.countries = self.hotels_df['country'].str.lower().unique().tolist()
            
            # Create name variants (without "The", "Hotel", etc.)
            self.hotel_name_variants = {}
            for idx, row in self.hotels_df.iterrows():
                original = row['hotel_name'].lower()
                # Create variants without common prefixes/suffixes
                variants = [
                    original,
                    re.sub(r'\bthe\b', '', original).strip(),
                    re.sub(r'\bhotel\b', '', original).strip(),
                    re.sub(r'\bthe\b|\bhotel\b', '', original).strip(),
                ]
                for variant in variants:
                    if variant:
                        self.hotel_name_variants[variant] = row['hotel_name']
            
        except FileNotFoundError:
            print("⚠️ CSV file not found. Using database-only extraction.")
            self.hotels_df = None
            self.hotel_names = []
            self.cities = []
            self.countries = []
            self.hotel_name_variants = {}
    
    def extract_hotel_name(self, query: str) -> Optional[str]:
        """
        Extract hotel name using multiple strategies with fuzzy matching.
        """
        query_lower = query.lower()
        
        # Strategy 1: Direct exact match in query
        for hotel_name in self.hotel_names:
            if hotel_name in query_lower:
                return self._get_original_name(hotel_name)
        
        # Strategy 2: Match variants (without "The", "Hotel", etc.)
        for variant, original in self.hotel_name_variants.items():
            if variant in query_lower and len(variant) > 3:
                return original
        
        # Strategy 3: Pattern matching for review queries
        patterns = [
            r'review[s]?\s+(?:for|of|about|on)\s+(.+?)(?:\s+hotel|\s+in\s+|\?|$)',
            r'(.+?)\s+review[s]?',
            r'(?:show|get|find)\s+(?:me\s+)?review[s]?\s+(?:for|of|about|on)\s+(.+?)(?:\s+hotel|\?|$)',
            r'what\s+(?:do\s+)?people\s+say\s+about\s+(.+?)(?:\s+hotel|\?|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential_name = match.group(1).strip()
                # Clean up common words
                potential_name = re.sub(r'\b(the|a|an|hotel)\b', '', potential_name).strip()
                
                if len(potential_name) > 2:
                    # Try fuzzy matching
                    matched_name = self._fuzzy_match_hotel(potential_name)
                    if matched_name:
                        return matched_name
        
        # Strategy 4: Fuzzy match individual words
        words = query_lower.split()
        for i in range(len(words)):
            for j in range(i+1, min(i+5, len(words)+1)):  # Check up to 4-word combinations
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3:
                    matched_name = self._fuzzy_match_hotel(phrase)
                    if matched_name:
                        return matched_name
        
        return None
    
    def _fuzzy_match_hotel(self, query_phrase: str, threshold: float = 0.6) -> Optional[str]:
        """
        Fuzzy match hotel names using similarity scoring.
        """
        if not self.hotel_names:
            return None
        
        # Try exact substring match first
        for hotel_name in self.hotel_names:
            if query_phrase in hotel_name or hotel_name in query_phrase:
                return self._get_original_name(hotel_name)
        
        # Use difflib for fuzzy matching
        matches = get_close_matches(query_phrase, self.hotel_names, n=1, cutoff=threshold)
        if matches:
            return self._get_original_name(matches[0])
        
        # Try matching against variants
        matches = get_close_matches(query_phrase, list(self.hotel_name_variants.keys()), n=1, cutoff=threshold)
        if matches:
            return self.hotel_name_variants[matches[0]]
        
        return None
    
    def _get_original_name(self, lowercase_name: str) -> str:
        """Get original hotel name with proper capitalization."""
        if self.hotels_df is not None:
            match = self.hotels_df[self.hotels_df['hotel_name'].str.lower() == lowercase_name]
            if not match.empty:
                return match.iloc[0]['hotel_name']
        return lowercase_name.title()
    
    def extract_city(self, query: str) -> Optional[str]:
        """Extract city from query with fuzzy matching."""
        query_lower = query.lower()
        
        # Direct match
        for city in self.cities:
            if city in query_lower:
                return self._get_original_city_name(city)
        
        # Fuzzy match
        words = query_lower.split()
        for word in words:
            if len(word) > 3:
                matches = get_close_matches(word, self.cities, n=1, cutoff=0.8)
                if matches:
                    return self._get_original_city_name(matches[0])
        
        return None
    
    def extract_country(self, query: str) -> Optional[str]:
        """Extract country from query with fuzzy matching."""
        query_lower = query.lower()
        
        # Direct match
        for country in self.countries:
            if country in query_lower:
                return self._get_original_country_name(country)
        
        # Fuzzy match for multi-word countries
        for country in self.countries:
            if ' ' in country:  # Multi-word countries
                if country in query_lower:
                    return self._get_original_country_name(country)
        
        return None
    
    def _get_original_city_name(self, lowercase_city: str) -> str:
        """Get original city name with proper capitalization."""
        if self.hotels_df is not None:
            match = self.hotels_df[self.hotels_df['city'].str.lower() == lowercase_city]
            if not match.empty:
                return match.iloc[0]['city']
        return lowercase_city.title()
    
    def _get_original_country_name(self, lowercase_country: str) -> str:
        """Get original country name with proper capitalization."""
        if self.hotels_df is not None:
            match = self.hotels_df[self.hotels_df['country'].str.lower() == lowercase_country]
            if not match.empty:
                return match.iloc[0]['country']
        return lowercase_country.title()
    
    def extract_all_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract all entities from query using CSV-enhanced matching.
        """
        entities = {
            'hotel_name': self.extract_hotel_name(query),
            'city': self.extract_city(query),
            'country': self.extract_country(query),
            'min_rating': None,
            'star_rating': None,
            'limit': 10
        }
        
        query_lower = query.lower()
        
        # Extract limit
        limit_match = re.search(r'(?:top|best|show|first|latest|recent)\s+(\d+)', query_lower)
        if limit_match:
            entities['limit'] = int(limit_match.group(1))
        
        # Extract star rating
        star_match = re.search(r'(\d+)\s*-?\s*star', query_lower)
        if star_match:
            entities['star_rating'] = int(star_match.group(1))
        
        # Extract minimum rating
        rating_match = re.search(r'rating\s+(?:>=?|above|over)\s*(\d+(?:\.\d+)?)', query_lower)
        if rating_match:
            entities['min_rating'] = float(rating_match.group(1))
        elif 'highly rated' in query_lower or 'top rated' in query_lower:
            entities['min_rating'] = 8.5
        
        return entities


# ============================================================================
# GRAPH-RAG SYSTEM (ENHANCED)
# ============================================================================

class GraphRAGSystem:
    """
    Enhanced Graph-RAG system with CSV-based entity extraction.
    """
    
    def __init__(self, uri: str, user: str, password: str, csv_path: str = "hotels.csv"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_models = {
            'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2'),
            'all-mpnet-base-v2': SentenceTransformer('all-mpnet-base-v2')
        }
        self.current_embedding_model = 'all-MiniLM-L6-v2'
        
        # ✨ Initialize CSV-based entity extractor
        self.csv_extractor = EnhancedEntityExtractor(csv_path)
        
        # Enhanced intent patterns with priority ordering
        self.intent_patterns = {
            'review_query': [  # HIGHER PRIORITY - Check first
                r'review[s]?\s+(?:for|of|about|on)',
                r'what\s+(?:do\s+)?people\s+say',
                r'feedback\s+(?:for|about|on)',
                r'rating[s]?\s+(?:for|of|about)',
                r'how\s+is\s+the',
                r'opinion[s]?\s+(?:on|about)',
                r'comment[s]?\s+(?:on|about)',
                r'what\s+(?:are\s+)?the\s+review[s]?',
                r'show\s+(?:me\s+)?review[s]?',
                r'get\s+review[s]?',
                r'find\s+review[s]?',
                r'customer\s+feedback',
                r'guest\s+(?:review|feedback|comment)'
            ],
            'hotel_info_query': [  # NEW: For "tell me about X hotel"
                r'tell\s+me\s+about',
                r'what\s+(?:is|are)\s+(?:the\s+)?(.+?)\s*\??',
                r'information\s+(?:about|on)',
                r'describe',
                r'details\s+(?:about|on|for)',
                r'know\s+(?:about|more)',
            ],
            'comparison_query': [
                r'compare\s+(.+?)\s+(?:and|vs|with)\s+(.+)',
                r'difference\s+between\s+(.+?)\s+and\s+(.+)',
                r'(.+?)\s+vs\s+(.+)',
                r'which\s+is\s+better.*?between\s+(.+?)\s+and\s+(.+)',
                r'comparing\s+(.+?)\s+(?:and|with)\s+(.+)',
            ],
            'traveller_specific': [
                r'(?:for|suitable for|best for|good for)\s+(?:family|families|kids|children)',
                r'(?:for|suitable for|best for|good for)\s+solo\s+travel(?:ler)?[s]?',
                r'(?:for|suitable for|best for|good for)\s+business\s+travel(?:ler)?[s]?',
                r'(?:for|suitable for|best for|good for)\s+couple[s]?',
                r'(?:for|suitable for|best for|good for)\s+romantic',
                r'family\s+friendly',
                r'business\s+friendly',
                r'romantic\s+hotel[s]?',
            ],
            'hotel_recommendation': [
                r'recommend\s+(?:me\s+)?(?:a\s+)?hotel[s]?',
                r'best\s+hotel[s]?',
                r'suggest\s+hotel[s]?',
                r'top\s+hotel[s]?',
                r'good\s+hotel[s]?',
                r'which\s+hotel',
                r'looking\s+for\s+(?:a\s+)?hotel'
            ],
            'hotel_search': [
                r'hotel[s]?\s+in\s+(\w+)',
                r'find\s+hotel[s]?',
                r'search\s+hotel[s]?',
                r'show\s+me\s+hotel[s]?',
                r'where\s+to\s+stay',
                r'accommodation[s]?\s+in',
                r'list\s+hotel[s]?'
            ],
            'visa_query': [
                r'visa\s+(?:requirement|required|need)',
                r'do\s+i\s+need\s+(?:a\s+)?visa',
                r'visa\s+(?:for|to)',
                r'entry\s+requirement[s]?'
            ],
            'statistics_query': [
                r'how\s+many',
                r'total\s+(?:number|count)',
                r'statistics',
                r'average',
                r'highest',
                r'lowest'
            ],
            'gender_query': [  # NEW: Gender-specific queries
                r'(?:for|suitable for|best for|good for)\s+(?:female|women|woman|ladies|girls)',
                r'(?:for|suitable for|best for|good for)\s+(?:male|men|man|gentleman|guys|boys)',
                r'female\s+(?:traveller|traveler|friendly)',
                r'male\s+(?:traveller|traveler|friendly)',
                r'women\s+(?:only|traveller|traveler|friendly)',
                r'men\s+(?:only|traveller|traveler|friendly)',
            ]
        }
        
        self.query_templates = self._initialize_query_templates()
        
    def _initialize_query_templates(self) -> Dict[str, str]:
        """Initialize Cypher query templates with fallback support."""
        return {
            # GENERIC queries (no city/country required)
            'all_hotels': """
                MATCH (h:Hotel)
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city, 
                       h.country AS country, h.star_rating AS star_rating,
                       h.average_reviews_score AS avg_score
                ORDER BY h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            # HOTEL INFO by name (NEW)
            'hotel_info_by_name': """
                MATCH (h:Hotel)
                WHERE toLower(h.name) CONTAINS toLower($hotel_name)
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                       h.country AS country, h.star_rating AS star_rating,
                       h.average_reviews_score AS avg_score,
                       h.cleanliness_base AS cleanliness, h.comfort_base AS comfort,
                       h.facilities_base AS facilities, h.location_base AS location,
                       h.staff_base AS staff, h.value_for_money_base AS value_for_money,
                       h.lat AS latitude, h.lon AS longitude
                LIMIT 1
            """,
            
            # HOTEL COMPARISON (NEW)
            'compare_hotels': """
                MATCH (h:Hotel)
                WHERE toLower(h.name) CONTAINS toLower($hotel1) 
                   OR toLower(h.name) CONTAINS toLower($hotel2)
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                       h.country AS country, h.star_rating AS star_rating,
                       h.average_reviews_score AS avg_score,
                       h.cleanliness_base AS cleanliness, h.comfort_base AS comfort,
                       h.facilities_base AS facilities, h.location_base AS location,
                       h.staff_base AS staff, h.value_for_money_base AS value_for_money
                ORDER BY h.name
            """,
            
            'top_rated_hotels_global': """
                MATCH (h:Hotel)
                WHERE h.average_reviews_score >= $min_rating
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                       h.country AS country, h.average_reviews_score AS avg_score
                ORDER BY h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            # REVIEW queries (enhanced)
            'reviews_by_hotel_name': """
                MATCH (h:Hotel)
                WHERE toLower(h.name) CONTAINS toLower($hotel_name)
                MATCH (h)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
                RETURN r.review_id AS review_id, r.text AS text, r.date AS date,
                       r.score_overall AS overall_score, r.score_cleanliness AS cleanliness,
                       r.score_comfort AS comfort, r.score_facilities AS facilities,
                       t.type AS traveller_type, t.age AS age, t.gender AS gender,
                       h.name AS hotel_name, h.city AS city
                ORDER BY r.date DESC
                LIMIT $limit
            """,
            
            'reviews_by_city': """
                MATCH (h:Hotel {city: $city})<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
                RETURN r.review_id AS review_id, r.text AS text, r.date AS date,
                       r.score_overall AS overall_score, t.type AS traveller_type,
                       h.name AS hotel_name, h.city AS city
                ORDER BY r.date DESC
                LIMIT $limit
            """,
            
            'recent_reviews_global': """
                MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
                RETURN r.review_id AS review_id, r.text AS text, r.date AS date,
                       r.score_overall AS overall_score, t.type AS traveller_type,
                       h.name AS hotel_name, h.city AS city, h.country AS country
                ORDER BY r.date DESC
                LIMIT $limit
            """,
            
            # City-specific queries
            'hotels_by_city': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city, 
                       h.country AS country, h.star_rating AS star_rating,
                       h.average_reviews_score AS avg_score,
                       h.cleanliness_base AS cleanliness, h.comfort_base AS comfort
                ORDER BY h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            'top_rated_hotels': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                WHERE h.average_reviews_score >= $min_rating
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                       h.average_reviews_score AS avg_score, h.star_rating AS star_rating
                ORDER BY h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            'hotels_by_star_rating': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                WHERE h.star_rating >= $min_stars
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.star_rating AS stars,
                       h.average_reviews_score AS avg_score
                ORDER BY h.star_rating DESC, h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            # Visa and statistics
            'visa_requirements': """
                MATCH (c1:Country {name: $from_country})-[v:NEEDS_VISA]->(c2:Country {name: $to_country})
                RETURN c1.name AS from_country, c2.name AS to_country,
                       v.requires AS requires_visa, v.visa_type AS visa_type
            """,
            
            'city_hotel_statistics': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                WITH c, 
                     COUNT(h) AS total_hotels,
                     AVG(h.average_reviews_score) AS avg_rating
                RETURN c.name AS city, total_hotels, avg_rating
            """
        }
    
    def classify_intent(self, query: str) -> List[str]:
        """Enhanced intent classification with priority ordering."""
        query_lower = query.lower()
        intents = []
        
        # PRIORITY 1: Check if hotel name is mentioned (hotel info query)
        # Pattern: "tell me about X" where X is a hotel name
        if re.search(r'tell\s+me\s+about', query_lower):
            intents.append('hotel_info_query')
        
        # PRIORITY 2: Check review_query
        for pattern in self.intent_patterns['review_query']:
            if re.search(pattern, query_lower):
                intents.append('review_query')
                break
        
        # PRIORITY 3: Check other intents
        for intent, patterns in self.intent_patterns.items():
            if intent in ['review_query', 'hotel_info_query']:  # Already checked
                continue
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intents.append(intent)
                    break
        
        # Default fallback - if we extracted a hotel name but no intent, assume hotel info
        if not intents:
            # This will be checked in extract_entities
            if any(word in query_lower for word in ['hotel', 'accommodation', 'stay', 'booking']):
                intents.append('hotel_search')
            else:
                intents.append('general_query')
        
        return intents
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Enhanced entity extraction using CSV data + database fallback.
        """
        # First try CSV-based extraction (better matching)
        csv_entities = self.csv_extractor.extract_all_entities(query)
        
        query_lower = query.lower()
        
        # Extract multiple hotel names for comparison queries
        hotel_names = self._extract_hotel_names_for_comparison(query_lower)
        
        # Combine with database extraction as fallback
        entities = {
            'hotel_name': csv_entities.get('hotel_name') or self._extract_hotel_name_db(query_lower),
            'hotel_names': hotel_names if hotel_names else [],  # For comparison queries
            'city': csv_entities.get('city') or self._extract_city_from_db(query_lower),
            'country': csv_entities.get('country') or self._extract_country_from_db(query_lower),
            'min_rating': csv_entities.get('min_rating'),
            'star_rating': csv_entities.get('star_rating'),
            'limit': csv_entities.get('limit', 10),
            'traveller_type': self._extract_traveller_type_from_db(query_lower),
            'gender': None,
            'age_group': None
        }
        
        # Extract gender
        if any(word in query_lower for word in ['female', 'women', 'woman']):
            entities['gender'] = 'Female'
        elif any(word in query_lower for word in ['male', 'men', 'man']):
            entities['gender'] = 'Male'
        
        
        return entities
    
    def _extract_hotel_names_for_comparison(self, query: str) -> List[str]:
        """Extract multiple hotel names for comparison queries."""
        hotel_names = []
        
        # Pattern: "compare X and Y" or "X vs Y"
        comparison_patterns = [
            r'compar(?:e|ing)\s+(.+?)\s+(?:and|vs|with)\s+(.+?)(?:\s+hotel[s]?|\?|$)',
            r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+hotel[s]?|\?|$)',
            r'(.+?)\s+vs\s+(.+?)(?:\s+hotel[s]?|\?|$)',
        ]
        
        for pattern in comparison_patterns:
            match = re.search(pattern, query.lower())
            if match:
                hotel1 = match.group(1).strip()
                hotel2 = match.group(2).strip()
                
                # Clean up common words
                hotel1 = re.sub(r'\b(the|a|an|hotel)\b', '', hotel1).strip()
                hotel2 = re.sub(r'\b(the|a|an|hotel)\b', '', hotel2).strip()
                
                if len(hotel1) > 2 and len(hotel2) > 2:
                    # Try fuzzy matching with CSV
                    matched1 = self.csv_extractor._fuzzy_match_hotel(hotel1)
                    matched2 = self.csv_extractor._fuzzy_match_hotel(hotel2)
                    
                    if matched1:
                        hotel_names.append(matched1)
                    else:
                        hotel_names.append(hotel1)
                    
                    if matched2:
                        hotel_names.append(matched2)
                    else:
                        hotel_names.append(hotel2)
                    
                    break
        
        return hotel_names
    
    def _extract_hotel_name_db(self, query_lower: str) -> Optional[str]:
        """Fallback: Extract hotel name from database."""
        match = re.search(r'review[s]?\s+(?:for|of|about|on)\s+(.+?)(?:\s+hotel|\s+in\s+|\?|$)', query_lower)
        if match:
            hotel_name = match.group(1).strip()
            hotel_name = re.sub(r'\b(the|a|an)\b', '', hotel_name).strip()
            if len(hotel_name) > 2:
                return hotel_name
        return None
    
    def _extract_city_from_db(self, query_lower: str) -> Optional[str]:
        """Extract city from database dynamically."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:City) RETURN DISTINCT c.name AS city")
                cities = [record['city'].lower() for record in result]
                for city in cities:
                    if city in query_lower:
                        return city.title()
        except:
            pass
        return None
    
    def _extract_country_from_db(self, query_lower: str) -> Optional[str]:
        """Extract country from database dynamically."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (co:Country) RETURN DISTINCT co.name AS country")
                countries = [record['country'].lower() for record in result]
                for country in countries:
                    if country in query_lower:
                        return country.title()
        except:
            pass
        return None
    
    def _extract_traveller_type_from_db(self, query_lower: str) -> Optional[str]:
        """Extract traveller type from database AND query patterns."""
        # First check common patterns
        traveller_type_patterns = {
            'Family': [r'\bfamily\b', r'\bfamilies\b', r'\bkids\b', r'\bchildren\b', r'\bfamily-friendly\b'],
            'Solo': [r'\bsolo\b', r'\balone\b', r'\bsolo travel', r'\bindividual\b'],
            'Business': [r'\bbusiness\b', r'\bwork\b', r'\bconference\b', r'\bcorporate\b'],
            'Couple': [r'\bcouple\b', r'\bromantic\b', r'\bhoneymoon\b', r'\bpartner\b']
        }
        
        for ttype, patterns in traveller_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return ttype
        
        # Fallback to database lookup
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (t:Traveller) RETURN DISTINCT t.type AS type")
                types = [record['type'].lower() for record in result if record['type']]
                for ttype in types:
                    if ttype in query_lower:
                        return ttype.title()
        except:
            pass
        return None
    
    def baseline_retrieval(self, query: str, intents: List[str], 
                          entities: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced baseline retrieval with better fallback handling."""
        results = {
            'cypher_queries': [],
            'data': [],
            'query_text': query,
            'fallback_used': False
        }
        
        with self.driver.session() as session:
            # PRIORITY 0: Hotel info query (NEW - "tell me about X hotel")
            if 'hotel_info_query' in intents and entities['hotel_name']:
                cypher = self.query_templates['hotel_info_by_name']
                params = {'hotel_name': entities['hotel_name']}
                results['cypher_queries'].append({'query': cypher, 'params': params})
                records = session.run(cypher, params)
                results['data'].extend([dict(record) for record in records])
            
            # PRIORITY 1: Review queries
            if 'review_query' in intents:
                if entities['hotel_name']:
                    cypher = self.query_templates['reviews_by_hotel_name']
                    params = {'hotel_name': entities['hotel_name'], 'limit': entities['limit']}
                    results['cypher_queries'].append({'query': cypher, 'params': params})
                    records = session.run(cypher, params)
                    results['data'].extend([dict(record) for record in records])
                    
                elif entities['city']:
                    cypher = self.query_templates['reviews_by_city']
                    params = {'city': entities['city'], 'limit': entities['limit']}
                    results['cypher_queries'].append({'query': cypher, 'params': params})
                    records = session.run(cypher, params)
                    results['data'].extend([dict(record) for record in records])
                    
                else:
                    cypher = self.query_templates['recent_reviews_global']
                    params = {'limit': entities['limit']}
                    results['cypher_queries'].append({'query': cypher, 'params': params})
                    records = session.run(cypher, params)
                    results['data'].extend([dict(record) for record in records])
                    results['fallback_used'] = True
            
            # PRIORITY 2: Hotel search/recommendation
            if 'hotel_search' in intents or 'hotel_recommendation' in intents:
                if entities['city']:
                    if entities['star_rating']:
                        cypher = self.query_templates['hotels_by_star_rating']
                        params = {
                            'city': entities['city'], 
                            'min_stars': entities['star_rating'],
                            'limit': entities['limit']
                        }
                    elif entities['min_rating']:
                        cypher = self.query_templates['top_rated_hotels']
                        params = {
                            'city': entities['city'],
                            'min_rating': entities['min_rating'],
                            'limit': entities['limit']
                        }
                    else:
                        cypher = self.query_templates['hotels_by_city']
                        params = {'city': entities['city'], 'limit': entities['limit']}
                    
                    results['cypher_queries'].append({'query': cypher, 'params': params})
                    records = session.run(cypher, params)
                    results['data'].extend([dict(record) for record in records])
                    
                else:
                    if entities['min_rating']:
                        cypher = self.query_templates['top_rated_hotels_global']
                        params = {'min_rating': entities['min_rating'], 'limit': entities['limit']}
                    else:
                        cypher = self.query_templates['all_hotels']
                        params = {'limit': entities['limit']}
                    
                    results['cypher_queries'].append({'query': cypher, 'params': params})
                    records = session.run(cypher, params)
                    results['data'].extend([dict(record) for record in records])
                    results['fallback_used'] = True
            
            # Statistics query
            if 'statistics_query' in intents:
                if entities['city']:
                    cypher = self.query_templates['city_hotel_statistics']
                    params = {'city': entities['city']}
                    results['cypher_queries'].append({'query': cypher, 'params': params})
                    records = session.run(cypher, params)
                    results['data'].extend([dict(record) for record in records])
            
            # PRIORITY 3: Comparison query (NEW)
            if 'comparison_query' in intents and entities.get('hotel_names') and len(entities['hotel_names']) >= 2:
                cypher = self.query_templates['compare_hotels']
                params = {
                    'hotel1': entities['hotel_names'][0],
                    'hotel2': entities['hotel_names'][1]
                }
                results['cypher_queries'].append({'query': cypher, 'params': params})
                records = session.run(cypher, params)
                results['data'].extend([dict(record) for record in records])
        
        return results
    
    def embedding_retrieval(self, query: str, entities: Dict[str, Any], 
                           model_name: str = 'all-MiniLM-L6-v2',
                           top_k: int = 5) -> Dict[str, Any]:
        """Semantic retrieval with fallback support."""
        model = self.embedding_models[model_name]
        query_embedding = model.encode(query).tolist()
        
        results = {'model': model_name, 'data': [], 'similarity_scores': []}
        
        with self.driver.session() as session:
            if entities['city']:
                cypher_query = """
                    MATCH (h:Hotel)
                    WHERE h.city = $city AND h.embedding IS NOT NULL
                    RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                           h.average_reviews_score AS avg_score, h.embedding AS embedding
                    LIMIT 100
                """
                params = {'city': entities['city']}
            else:
                cypher_query = """
                    MATCH (h:Hotel)
                    WHERE h.embedding IS NOT NULL
                    RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                           h.average_reviews_score AS avg_score, h.embedding AS embedding
                    LIMIT 100
                """
                params = {}
            
            records = session.run(cypher_query, params)
            hotels = [dict(record) for record in records]
            
            if not hotels:
                return results
            
            embeddings_matrix = np.array([h['embedding'] for h in hotels])
            query_embedding_array = np.array(query_embedding).reshape(1, -1)
            similarities = cosine_similarity(query_embedding_array, embeddings_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                hotel = hotels[idx]
                results['data'].append({
                    'hotel_id': hotel['hotel_id'],
                    'name': hotel['name'],
                    'city': hotel['city'],
                    'avg_score': hotel['avg_score'],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def combined_retrieval(self, query: str, retrieval_method: str = "hybrid") -> Dict[str, Any]:
        """
        Combined retrieval with user-configurable strategy.
        
        Args:
            query: User query string
            retrieval_method: "baseline", "embeddings", or "hybrid" (default: "hybrid")
        
        Returns:
            Dictionary containing query results and metadata
        """
        # Validate retrieval method
        valid_methods = ["baseline", "embeddings", "hybrid"]
        if retrieval_method not in valid_methods:
            print(f"⚠️ Invalid method '{retrieval_method}', defaulting to 'hybrid'")
            retrieval_method = "hybrid"
        
        intents = self.classify_intent(query)
        entities = self.extract_entities(query)
        
        # SMART FALLBACK: If we have a hotel name but no clear intent, add hotel_info_query
        if entities.get('hotel_name') and 'general_query' in intents:
            intents = ['hotel_info_query'] + intents
        
        print(f"🎯 Detected intents: {intents}")
        print(f"📍 Extracted entities: {entities}")
        print(f"🔧 Retrieval method: {retrieval_method}")
        
        # Initialize results
        baseline_results = {
            'data': [], 
            'cypher_queries': [], 
            'query_text': query,
            'fallback_used': False
        }
        embedding_results = None
        
        # Execute baseline retrieval
        if retrieval_method in ["baseline", "hybrid"]:
            baseline_results = self.baseline_retrieval(query, intents, entities)
            print(f"✓ Baseline retrieved {len(baseline_results['data'])} results")
        
        # Execute embedding retrieval
        if retrieval_method in ["embeddings", "hybrid"]:
            # Only use embeddings for appropriate intents
            if 'hotel_search' in intents or 'hotel_recommendation' in intents:
                try:
                    embedding_results = self.embedding_retrieval(
                        query, entities, self.current_embedding_model
                    )
                    if embedding_results:
                        print(f"✓ Embeddings retrieved {len(embedding_results['data'])} results")
                except Exception as e:
                    print(f"❌ Embedding retrieval failed: {e}")
                    
                    # CRITICAL: If embeddings-only mode fails, fallback to baseline
                    if retrieval_method == "embeddings" and not baseline_results['data']:
                        print("⚠️ Embeddings-only mode failed, falling back to baseline")
                        baseline_results = self.baseline_retrieval(query, intents, entities)
                        baseline_results['fallback_used'] = True
            else:
                print(f"ℹ️ Intent '{intents}' not suitable for embedding search")
                
                # If embeddings-only but intent doesn't support it, use baseline
                if retrieval_method == "embeddings":
                    print("⚠️ Query type not suitable for embeddings, falling back to baseline")
                    baseline_results = self.baseline_retrieval(query, intents, entities)
                    baseline_results['fallback_used'] = True
        
        # Construct combined result
        combined = {
            'query': query,
            'intents': intents,
            'entities': entities,
            'retrieval_method': retrieval_method,
            'baseline': baseline_results,
            'embedding': embedding_results,
            'combined_data': []
        }
        
        # Merge results with deduplication
        seen_ids = set()
        
        # Priority 1: Add baseline results (if baseline or hybrid mode)
        if retrieval_method in ["baseline", "hybrid"]:
            for item in baseline_results['data']:
                item_id = item.get('hotel_id') or item.get('review_id')
                if item_id and item_id not in seen_ids:
                    item['source'] = 'baseline'  # Tag source for debugging
                    combined['combined_data'].append(item)
                    seen_ids.add(item_id)
        
        # Priority 2: Add embedding results (if embeddings or hybrid mode)
        if retrieval_method in ["embeddings", "hybrid"] and embedding_results:
            for item in embedding_results['data']:
                if item['hotel_id'] not in seen_ids:
                    item['source'] = 'embeddings'  # Tag source for debugging
                    combined['combined_data'].append(item)
                    seen_ids.add(item['hotel_id'])
        
        # Final summary
        total_results = len(combined['combined_data'])
        print(f"📊 Total results: {total_results}")
        if baseline_results.get('fallback_used'):
            print("⚠️ Note: Fallback to baseline was triggered")
        
        return combined
    
    def create_node_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Create embeddings for hotel nodes."""
        model = self.embedding_models[model_name]
        
        with self.driver.session() as session:
            records = session.run("""
                MATCH (h:Hotel)
                RETURN h.hotel_id AS hotel_id, h.name AS name, h.city AS city,
                       h.country AS country, h.star_rating AS stars,
                       h.average_reviews_score AS avg_score
            """)
            
            hotels = [dict(record) for record in records]
            for hotel in hotels:
                description = f"{hotel['name']} is a {hotel['stars']}-star hotel in {hotel['city']}, {hotel['country']}. "
                description += f"Average rating: {hotel['avg_score']:.1f}."
                embedding = model.encode(description).tolist()
                
                session.run("""
                    MATCH (h:Hotel {hotel_id: $hotel_id})
                    SET h.embedding = $embedding
                """, {'hotel_id': hotel['hotel_id'], 'embedding': embedding})
        
        print(f"✅ Node embeddings created using {model_name}")
    
    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()