# app.py
import streamlit as st
from openai import OpenAI
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set
from math import radians, sin, cos, sqrt, atan2
import logging

# Yr openai key here
openai_key = ""  

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Setup ---
st.set_page_config(page_title="MatchAI", page_icon="🤖", layout="wide")
st.title("🤖 MatchAI — Offers Chatbot")

# --- Constants ---
DISTRICT_TO_COORDS = {
    "orchard": [1.3048, 103.8318],
    "tanjong pagar": [1.276, 103.844],
    "marina bay": [1.284, 103.859],
    "jurong": [1.26, 103.67],
    "holland": [1.312, 103.793],
    "city hall": [1.293, 103.853],
    "clarke quay": [1.2905, 103.8468],
    "changi": [1.3592, 103.989],
    "dempsey": [1.306, 103.805],
    "tampines": [1.3546, 103.9458],
    "sentosa": [1.2543, 103.8239],
    "bugis": [1.299, 103.857],
    "geylang": [1.309, 103.892],
    "raffles place": [1.283, 103.851],
    "kallang": [1.3023, 103.864],
    "katong": [1.308, 103.901],
    "little india": [1.305, 103.855],
    "kampong glam": [1.300, 103.859],
    "harbourfront": [1.265, 103.822],
    "tanglin": [1.307, 103.816],
    "mandai": [1.4042, 103.7885],
}

# --- Helper Functions ---
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filename):
        st.error(f"Data file {filename} not found. Please ensure the file exists and is populated.")
        return []
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

@st.cache_data
def load_data():
    merchants = load_jsonl('merchants_sg.jsonl')
    customers = load_jsonl('customers_sg.jsonl')
    if not merchants or not customers:
        st.warning("No merchant or customer data loaded. Please check your data files.")
    # Normalize keys to ints when possible (so selection indices and dict lookups are consistent)
    merchants_dict = {}
    for m in merchants:
        key = m.get('merchant_id', None)
        try:
            key = int(key)
        except Exception:
            pass
        if key is None:
            continue
        merchants_dict[key] = m

    customers_dict = {}
    for c in customers:
        key = c.get('consumer_id', None)
        try:
            key = int(key)
        except Exception:
            pass
        if key is None:
            continue
        customers_dict[key] = c

    return merchants_dict, customers_dict

merchants_dict, customers_dict = load_data()

def parse_price_range(price_range: str) -> Tuple[float, float]:
    # Handle the case where price_range might be a dictionary from the refinement step
    if isinstance(price_range, dict):
        # Assume the dictionary has 'min' and 'max' keys
        min_val = price_range.get('min', 0)
        max_val = price_range.get('max', float('inf'))
        # Convert to float in case they are strings within the dict
        try:
            min_val = float(min_val)
            max_val = float(max_val)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert dict price range {price_range} to float, defaulting to (0, inf)")
            return 0, float('inf')
        return min_val, max_val

    # Original string handling logic
    if not price_range:
        return 0, float('inf')
    price_range = price_range.lower().replace('sgd', '').strip()
    if 'around' in price_range or 'about' in price_range:
        import re
        match = re.search(r'around\s*(\d+(?:\.\d+)?)|about\s*(\d+(?:\.\d+)?)', price_range)
        if match:
            center = float(match.group(1) or match.group(2))
            return max(0, center - 10), center + 10  # +/- 10 SGD range
    if price_range == "cheap":
        return 0, 20
    elif price_range == "mid-range":
        return 20, 100
    elif price_range == "expensive":
        return 100, float('inf')
    elif "under $" in price_range:
        try:
            max_price = float(price_range.replace("under $", ""))
            return 0, max_price
        except Exception:
            return 0, float('inf')
    elif "$" in price_range:
        parts = price_range.replace("$", "").split("-")
        try:
            min_price = float(parts[0])
            max_price = float(parts[1]) if len(parts) > 1 and parts[1].strip() != "" else float('inf')
            return min_price, max_price
        except Exception:
            return 0, float('inf')
    else:
        try:
            price = float(price_range)
            return max(0, price - 10), price + 10
        except ValueError:
            return 0, float('inf')

# --- OpenAI client helper ---
def get_openai_client():
    api_key = openai_key
    if not api_key:
        st.error("OpenAI API key not set. Please add it to the code.")
        raise RuntimeError("OpenAI API key not set")
    client = OpenAI(api_key=api_key)
    return client

# --- Intent extraction and preference deduction (with safer, defensive API calls) ---
def extract_intent(user_request: str, model: str = None) -> Dict[str, Any]:
    # Use gpt-5 instead of defaulting to gpt-3.5-turbo
    model = "gpt-5-nano" # or os.getenv("OPENAI_MODEL", "gpt-5") if you want it configurable via env var
    intent_prompt = f"""
    Analyze the user request: "{user_request}"

    Extract the following in JSON format:
    {{
        "category": "travel|food|activities|gifts|general",
        "subcategory": "str or null",
        "exact_items": ["str", ...],
        "price_range": "str or null",
        "proximity_km": null,
        "min_rating": null,
        "specifics": {{
            "destination": null,
            "mode": null,
            "duration": null,
            "participants": null,
            "recipient": null,
            "cuisine": null,
            "atmosphere": null,
            "delivery": null,
            "legal": null,
            "bundle": false,
            "solo": false,
            "type": null
        }},
        "is_generic": false,
        "is_ambiguous": false,
        "confidence": "high"
    }}

    Rules:
    - Respond only with a JSON object.
    - If you can't determine a field, set it to null or false (as appropriate).
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": intent_prompt}],
            # Changed: Use max_completion_tokens instead of max_tokens
        )
        # Defensive parsing
        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            try:
                content = response.choices[0].message.content.strip()
            except Exception:
                # some SDK versions expose message as dict
                content = response.choices[0].message.get("content", "").strip() if isinstance(response.choices[0].message, dict) else ""
        if not content:
            logger.warning("Empty intent extraction response from OpenAI")
            return {}

        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            # fallback: try to extract JSON substring
            import re
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}
    except Exception as e:
        logger.exception("OpenAI error in extract_intent")
        return {}

# --- Refine Intent based on User Profile and Real Intent ---
def refine_intent_with_real_intent(user_request: str, initial_intent: Dict[str, Any], customer: Dict[str, Any], model: str = "gpt-5-nano") -> Dict[str, Any]:
    """
    Uses an LLM to interpret the real intent from the user's request and profile,
    potentially refining the initial intent extracted by extract_intent.
    """
    profile_str = f"""
    Customer Profile:
    - Name: {customer.get('name')}
    - Age: {customer.get('age')}
    - Sex: {customer.get('sex')}
    - Occupation: {customer.get('occupation')}
    - Balance: ${customer.get('balance', 0):.2f}
    - Location: {customer.get('address', {}).get('district')}
    - Recent Transactions: {[(t.get('merchant_id'), t.get('category'), t.get('amount')) for t in customer.get('recent_transactions', [])]}
    """

    refine_prompt = f"""
    Original User Request: "{user_request}"
    Initial Intent: {json.dumps(initial_intent, indent=2)}
    {profile_str}

    Based on the user's request, the initial intent, and their profile, analyze the *real intent*.
    Consider:
    - What is the user *actually* looking for beyond the literal words?
    - What context might be missing from the initial intent?
    - What might be implied by their location, recent spending, or the phrasing itself (e.g., "near me", "activities", "family")?

    Output a refined JSON intent object with the same structure as the initial intent, potentially updating fields like:
    - category (e.g., "general" might become "food" or "activities")
    - subcategory
    - exact_items (only include if the user explicitly asked for specific items like "sushi" or "chocolate cake")
    - price_range (e.g., "estimate based on profile balance if budget or range not specified")
    - proximity_km 
    - min_rating
    - specifics (e.g., "near me" -> specifics.proximity_km, "family activities" -> specifics.participants = "family")
    - is_generic (e.g., "family activities" is specific enough to be False)
    - is_ambiguous (e.g., "activities" is ambiguous, "family activities" is less so)
    - confidence

    Rules:
    - Respond only with a JSON object matching the intent structure.
    - Prioritize specific, actionable intent over generic placeholders.
    - If the request implies a common default (like proximity), set it explicitly.
    - Only include items in 'exact_items' if they were explicitly mentioned by the user or are strongly implied as specific physical items/services. Avoid adding conceptual items like "afternoon_tea_for_two_voucher" unless the user asked for "a voucher for afternoon tea".
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": refine_prompt}],
            # Changed: Use max_completion_tokens instead of max_tokens
        )
        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            try:
                content = response.choices[0].message.content.strip()
            except Exception:
                content = response.choices[0].message.get("content", "").strip() if isinstance(response.choices[0].message, dict) else ""
        if not content:
            logger.warning("Empty intent refinement response from OpenAI")
            return initial_intent # Fallback to initial intent if refinement fails

        try:
            parsed = json.loads(content)
            logger.info(f"Refined Intent: {parsed}") # Log for debugging
            return parsed
        except json.JSONDecodeError:
            # fallback: try to extract JSON substring
            import re
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return initial_intent # Fallback if JSON parsing fails
            return initial_intent # Fallback if no JSON found
    except Exception as e:
        logger.exception("OpenAI error in refine_intent_with_real_intent")
        return initial_intent # Fallback to initial intent if API call fails


def deduce_profile_preferences(customer: Dict[str, Any], category: str, model: str = None) -> Set[str]:
    # Use gpt-5 instead of defaulting to gpt-3.5-turbo
    model = "gpt-5-nano" # or os.getenv("OPENAI_MODEL", "gpt-5") if you want it configurable via env var
    # CRITICAL FIX: Use {} instead of {{}} in the f-string for address
    profile_prompt = f"""
    Deduce preferences for a {category} request based on the following user profile:

    Name: {customer.get('name')}
    Age: {customer.get('age')}
    Sex: {customer.get('sex')}
    Occupation: {customer.get('occupation')}
    Balance: ${customer.get('balance', 0):.2f}
    Location: {customer.get('address', {}).get('district')} # Fixed: Corrected {{}} to {{}}
    Recent Transactions: {[(t.get('merchant_id'), t.get('category'), t.get('amount')) for t in customer.get('recent_transactions', [])]}

    Output a JSON list of likely preferences for {category}, e.g. ["japanese", "fine dining"].
    Respond ONLY with valid JSON list.
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": profile_prompt}],
            # Changed: Use max_completion_tokens instead of max_tokens
        )
        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            try:
                content = response.choices[0].message.content.strip()
            except Exception:
                content = response.choices[0].message.get("content", "").strip() if isinstance(response.choices[0].message, dict) else ""
        if not content:
            return set()
        try:
            prefs = json.loads(content)
            return set(prefs if isinstance(prefs, list) else [])
        except json.JSONDecodeError:
            return set()
    except Exception:
        logger.exception("OpenAI error in deduce_profile_preferences")
        return set()

def find_relevant_merchants(
    customer: Dict[str, Any],
    customer_coords: List[float],
    intent: Dict[str, Any],
    top_k: int = 10,
) -> List[Tuple[Dict[str, Any], Optional[float]]]:
    scored_merchants = []
    # Get exact_items from intent
    exact_items = intent.get('exact_items', []) if intent else []
    has_specific_items = bool(exact_items)  # True if list is not empty

    price_min, price_max = parse_price_range(intent.get('price_range')) if intent else (0, float('inf'))
    proximity_km = intent.get('proximity_km') if intent else None
    min_rating = intent.get('min_rating') if intent else None
    category = intent.get('category', 'general') if intent else 'general'
    specifics = intent.get('specifics', {}) if intent else {}
    is_generic = intent.get('is_generic', False) if intent else False
    is_ambiguous = intent.get('is_ambiguous', False) if intent else False
    confidence = intent.get('confidence', 'medium') if intent else 'medium'

    # --- Deduce preferences for generic/ambiguous requests ---
    preferences = set()
    if is_generic or is_ambiguous or confidence == 'low':
        preferences = deduce_profile_preferences(customer, category)

    for mid, merchant in merchants_dict.items():
        score = 0.0
        dist = None

        # --- Distance filter ---
        try:
            if proximity_km is not None and customer_coords:
                m_lat = float(merchant.get('lat', 1.29))
                m_lon = float(merchant.get('lon', 103.85))
                dist = haversine(customer_coords[0], customer_coords[1], m_lat, m_lon)
                if dist > proximity_km:
                    continue
        except Exception:
            # if merchant lat/lon missing, skip distance filter
            dist = None

        # --- Rating filter ---
        try:
            if min_rating and float(merchant.get('rating', 0.0)) < float(min_rating):
                continue
        except Exception:
            pass

        # --- Price filter using DISCOUNTED price ---
        items = merchant.get('items', [])
        if not items:
            continue

        in_budget = False
        for item in items:
            price = item.get('price')
            disc_pct = item.get('discount_percent', 0)

            # Validate and sanitize price
            if not isinstance(price, (int, float)) or price < 0:
                continue

            # Validate and sanitize discount_percent
            if not isinstance(disc_pct, (int, float)) or disc_pct < 0 or disc_pct > 100:
                disc_pct = 0

            discounted_price = price * (1 - disc_pct / 100)

            if price_min <= discounted_price <= price_max:
                in_budget = True
                break

        if not in_budget:
            continue

        # --- Category/subcategory matching (Primary Strategy) ---
        merchant_category = (merchant.get('category') or "").lower()
        primary_category = (merchant.get('primary_category') or "").lower()

        category_match = False
        if category != "general":
            if category.lower() in primary_category:
                score += 2
                category_match = True
            elif category.lower() in merchant_category:
                score += 1
                category_match = True
            # Recipient-based logic for gifts
            if not category_match:
                if category == "gifts":
                    recipient = (specifics.get('recipient', '') or '').lower()
                    if recipient == "mother":
                        mother_keywords = ["florist", "spa", "jewelry", "chocolates", "books", "fashion", "souvenirs", "stationery", "e-gift cards"]
                        if any(keyword in primary_category or keyword in merchant_category for keyword in mother_keywords):
                            score += 2
                            category_match = True
        else:
            category_match = True  # Allow all for 'general'

        if not category_match:
            continue

        # --- Specific item matching (only if user requested exact items) ---
        if has_specific_items:
            merchant_items = [item.get('name', '').lower() for item in items]
            if not all(any(i.lower() in mi for mi in merchant_items) for i in exact_items):
                continue
            score += 3

        # --- Additional specifics matching (refinements) ---
        if category == "food":
            cuisine = (specifics.get('cuisine', '') or '').lower()
            atmosphere = (specifics.get('atmosphere', '') or '').lower()
            if cuisine and cuisine not in merchant_category and cuisine not in primary_category:
                score -= 2
            if atmosphere and atmosphere not in merchant_category:
                score -= 1

        elif category == "activities":
            participants = (specifics.get('participants', '') or '').lower()
            solo = specifics.get('solo', False)
            activity_type = (specifics.get('type', '') or '').lower()
            if solo and 'solo' not in merchant_category and 'individual' not in primary_category:
                score -= 1
            if activity_type == 'sports':
                if not any(sport in merchant_category for sport in ['sports', 'swimming', 'gym', 'fitness', 'running']):
                    score -= 2
            if participants == "couple":
                if not any(cat in merchant_category for cat in ["spa", "romantic", "fine dining"]):
                    score -= 2
            elif participants == "family":
                family_keywords = ["wildlife", "theme park", "park", "hawker", "family", "playground", "kids", "workshop"]
                if not any(keyword in merchant_category or keyword in primary_category for keyword in family_keywords):
                    score -= 2

        # --- Boost for deduced preferences (only for ambiguous/generic requests without specific items) ---
        if (is_generic or is_ambiguous or confidence == 'low') and preferences and not has_specific_items:
            if any(pref in merchant_category or pref in primary_category for pref in preferences):
                score += 1

        # --- Bonus for proximity ---
        if dist is not None:
            score += max(0, 5 / (dist + 1))

        if score > 0:
            scored_merchants.append((score, merchant, dist))

    # Sort by score descending and return top_k
    scored_merchants.sort(key=lambda x: x[0], reverse=True)
    return [(m, d) for _, m, d in scored_merchants[:top_k]]

SYSTEM_PROMPT = """
You are a personalized offer chatbot for a Singapore rewards program.
You recommend deals from local merchants based on the user's request and profile.

User Profile: {user_profile}
Intent: {intent}
Relevant Merchants: {relevant_merchants}

Rules:
- Only recommend merchants from the list above.
- For specific requests, only recommend merchants that match all criteria.
- For generic/ambiguous requests, use the deduced preferences.
- Highlight discounts, ratings, proximity, and fit to the request.
- When mentioning prices, clearly state the original price, discount percentage, and discounted price in a readable format, e.g., "originally $48.00, now $43.20 after 10% off".
- Do not concatenate numbers or create garbled text; ensure clear spacing and punctuation.
- If no suitable merchants are found, say so and suggest broadening the search.
- Suggest 2-3 offers with clear reasons.
- Keep responses friendly, concise, and engaging.
- Start your response with "Recommended Offers" followed by the recommendations.
"""

def generate_personalized_offer(customer_id: int, user_request: str) -> Tuple[str, List[Tuple[Dict[str, Any], Optional[float]]], Dict[str, Any]]:
    # Normalize customer_id lookup
    try:
        cid_key = int(customer_id)
    except Exception:
        cid_key = customer_id

    if cid_key not in customers_dict:
        return "Customer not found. Please provide a valid customer ID.", [], {}

    customer = customers_dict[cid_key]
    district = customer.get('address', {}).get('district', '') or ''
    customer_coords = DISTRICT_TO_COORDS.get(district.lower(), [1.29, 103.85])

    # --- Extract initial intent ---
    initial_intent = extract_intent(user_request)
    if not initial_intent:
        initial_intent = {"category": "general", "is_generic": True, "confidence": "low", "specifics": {}}

    # --- Refine intent based on real intent and profile ---
    refined_intent = refine_intent_with_real_intent(user_request, initial_intent, customer)
    # Use the refined intent for further processing
    intent_to_use = refined_intent if refined_intent else initial_intent

    # --- Find relevant merchants ---
    relevant_list = find_relevant_merchants(customer, customer_coords, intent_to_use)

    if not relevant_list:
        return "No merchants found that match your request. Try broadening your search criteria!", [], intent_to_use

    # --- Generate merchant summary for system prompt ---
    merchants_summary = ""
    for merchant, dist in relevant_list:
        dist_str = f" ({dist:.1f}km away)" if dist is not None else ""
        merchants_summary += f"- {merchant.get('name')} ({merchant.get('primary_category')}, Rating: {merchant.get('rating')}{dist_str}):\n"
        for item in merchant.get('items', [])[:3]:  # Show top 3 items
            try:
                price = float(item.get('price', 0.0))
                disc_pct = float(item.get('discount_percent', 0.0))
                discounted_price = price * (1 - disc_pct / 100)
                merchants_summary += f"  - {item.get('name')}: Original ${price:.2f}, Discounted ${discounted_price:.2f} (with {disc_pct:.0f}% off)\n"
            except Exception:
                continue

    user_profile = f"""
    Name: {customer.get('name')}
    Age: {customer.get('age')}, Sex: {customer.get('sex')}
    Occupation: {customer.get('occupation')}
    Balance: ${customer.get('balance', 0):.2f}
    Location: {customer.get('address', {}).get('district')}
    Recent Transactions: {[(t.get('merchant_id'), t.get('category')) for t in customer.get('recent_transactions', [])]}
    """

    full_prompt = SYSTEM_PROMPT.format(
        user_profile=user_profile,
        intent=json.dumps(intent_to_use, indent=2),
        relevant_merchants=merchants_summary
    )

    try:
        client = get_openai_client()
        # Use gpt-5 instead of defaulting to gpt-3.5-turbo
        model = "gpt-5-nano" # or os.getenv("OPENAI_MODEL", "gpt-5") if you want it configurable via env var
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_request}
            ],
            # Changed: Use max_completion_tokens instead of max_tokens
        )
        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            try:
                content = response.choices[0].message.content.strip()
            except Exception:
                content = response.choices[0].message.get("content", "").strip() if isinstance(response.choices[0].message, dict) else ""
        if not content:
            logger.warning("Empty offer response from OpenAI")
            return "Couldn't generate an offer at the moment. Please try again.", relevant_list, intent_to_use

        return content, relevant_list, intent_to_use

    except Exception as e:
        logger.exception("Error calling OpenAI in generate_personalized_offer")
        return f"Error generating offer: {str(e)}", relevant_list, intent_to_use

# --- State Management ---
if 'app_settings' not in st.session_state:
    st.session_state.app_settings = {"affinity_priority": True, "show_raw": False}
if 'selected_customer_option' not in st.session_state:
    st.session_state.selected_customer_option = None

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.session_state.app_settings["affinity_priority"] = st.checkbox("Prioritize Affinity (Past Transactions)", value=st.session_state.app_settings["affinity_priority"])
    st.session_state.app_settings["show_raw"] = st.checkbox("Show Raw Data", value=st.session_state.app_settings["show_raw"])

    customer_options = {}
    for cid, customer in customers_dict.items():
        district = customer.get('address', {}).get('district', '')
        balance = f"${customer.get('balance', 0):.0f}"
        option_text = f"{cid}: {customer.get('name')} ({customer.get('age')} {customer.get('sex')}, {district}, {balance})"
        customer_options[option_text] = cid

    if not customer_options:
        st.error("No customer data available. Please check your data files.")
        st.stop()

    if st.session_state.selected_customer_option is None:
        st.session_state.selected_customer_option = list(customer_options.keys())[0]

    selected_option = st.selectbox("Select Customer", options=list(customer_options.keys()), index=list(customer_options.keys()).index(st.session_state.selected_customer_option))
    st.session_state.selected_customer_option = selected_option
    customer_id = customer_options[selected_option]

# --- Main Tabs ---
tabs = st.tabs(["💬 Generate", "📜 History", "📊 Merchant Exposure"])

with tabs[0]:
    st.write("Welcome! Tell us what you're looking for.")
    user_request = st.text_input("What are you looking for today?")

    if st.button("Get Offers"):
        if user_request:
            with st.spinner("Finding the best offers for you..."):
                try:
                    offer, relevant_merchants, intent = generate_personalized_offer(customer_id, user_request)
                    # Ensure recommended_merchant_ids is always a list, even if relevant_merchants is empty
                    recommended_merchant_ids = [merchant.get('merchant_id') for merchant, _ in relevant_merchants if merchant.get('merchant_id') is not None]
                except RuntimeError:
                    # Missing API key or fatal error already reported to user
                    offer = "OpenAI API key missing or error. Check logs."
                    relevant_merchants = []
                    intent = {}
                    recommended_merchant_ids = [] # Default to empty list on error

                st.subheader("Recommended Offers")
                st.write(offer)

                if relevant_merchants:
                    st.subheader("Reference Merchants (Original Data)")
                    for i, (merchant, dist) in enumerate(relevant_merchants, 1):
                        with st.expander(f"Merchant {i}: {merchant.get('name')} (ID: {merchant.get('merchant_id')})"):
                            if dist is not None:
                                st.write(f"**Distance:** {dist:.1f} km")
                            if st.session_state.app_settings["show_raw"]:
                                st.json(merchant)

                # --- Log conversation (ENSURE ALL KEYS PRESENT)---
                try:
                    log_data = {
                        'timestamp': [datetime.now().isoformat()],
                        'customer_id': [customer_id],
                        'user_request': [user_request],
                        'intent': [json.dumps(intent)],
                        'offer': [offer],
                        'recommended_merchants': [json.dumps(recommended_merchant_ids)] # Always present
                    }
                    df_new = pd.DataFrame(log_data)
                    csv_file = 'conversations.csv'
                    try:
                        df_existing = pd.read_csv(csv_file)
                        # Ensure the existing df has the column (in case it was created incorrectly before)
                        if 'recommended_merchants' not in df_existing.columns:
                             df_existing['recommended_merchants'] = '[]' # Default value
                        df = pd.concat([df_existing, df_new], ignore_index=True)
                    except FileNotFoundError:
                        # File doesn't exist yet, df_new becomes the initial df
                        df = df_new
                    df.to_csv(csv_file, index=False)
                    st.success(f"Conversation logged to {csv_file}")
                except Exception:
                    logger.exception("Failed to log conversation")
        else:
            st.warning("Please enter a request.")

with tabs[1]:
    st.subheader("Conversation History")
    csv_file = 'conversations.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        st.dataframe(df)
        if st.button("Clear History"):
            try:
                os.remove(csv_file)
                st.success("History cleared!")
                st.rerun() # Refresh the page to reflect the change
            except OSError as e:
                st.error(f"Error deleting file: {e}")
    else:
        st.info("No conversation history yet.")

with tabs[2]:
    st.subheader("Merchant Exposure Analytics")
    csv_file = 'conversations.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if 'recommended_merchants' not in df.columns:
            st.warning("The 'recommended_merchants' column is missing from the conversation log. "
                       "This might be due to an older log file format or an issue during initial logging. "
                       "Generating new offers should add this column.")
        else:
            # --- Rest of the analytics logic ---
            def safe_load(v):
                try:
                    return json.loads(v) if isinstance(v, str) else v
                except Exception:
                    return [] # Return empty list on JSON parse error

            df['recommended_merchants'] = df['recommended_merchants'].apply(safe_load)
            exploded_df = df.explode('recommended_merchants')
            # Filter out rows where the value is an empty list or NaN after explode
            exploded_df = exploded_df.dropna(subset=['recommended_merchants'])
            exploded_df = exploded_df[exploded_df['recommended_merchants'].apply(lambda x: x != [])] # Also drop empty lists if they slipped through

            if not exploded_df.empty:
                exposure_counts = exploded_df['recommended_merchants'].value_counts().reset_index()
                exposure_counts.columns = ['Merchant ID', 'Exposure Count']
                
                # Merge with merchant details to create the leaderboard table
                leaderboard_data = []
                for _, row in exposure_counts.iterrows():
                    mid = row['Merchant ID']
                    exp_count = row['Exposure Count']
                    merchant_info = merchants_dict.get(mid, {})
                    leaderboard_data.append({
                        "Merchant ID": mid,
                        "Name": merchant_info.get("name", "Unknown"),
                        "Primary Category": merchant_info.get("primary_category", "N/A"),
                        "Rating": merchant_info.get("rating", "N/A"),
                        "District": merchant_info.get("address", {}).get("district", "N/A"),
                        "Exposure Count": exp_count
                    })
                
                if leaderboard_data: # Check if the list is not empty
                    leaderboard_df = pd.DataFrame(leaderboard_data)
                    # Sort by Exposure Count descending
                    leaderboard_df = leaderboard_df.sort_values(by="Exposure Count", ascending=False)
                    # Changed: Use width='stretch' instead of use_container_width=True
                    st.dataframe(leaderboard_df, width='stretch')
                else:
                    st.info("No valid merchant IDs found in the conversation history for analytics.")
            else:
                st.info("No valid merchant recommendations to display in analytics.")
        
        if st.button("Clear Analytics"):
            try:
                os.remove(csv_file)
                st.success("Analytics cleared! (This also clears history)")
                st.rerun() # Refresh the page to reflect the change
            except OSError as e:
                st.error(f"Error deleting file: {e}")
    else:
        st.info("No conversation history file found yet.")
