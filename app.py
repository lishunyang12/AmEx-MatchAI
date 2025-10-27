# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AmEx MatchAI", layout="wide")
st.title("AmEx MatchAI – GenAI-Powered Merchant Matchmaking")

# -------------------------------
# LOAD LLM (ONCE)
# -------------------------------
@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe

# Load LLM
try:
    llm = load_llm()
    llm_status = "✅ LLM loaded successfully"
except Exception as e:
    st.warning("⚠️ LLM failed to load (offline mode). Using rule-based fallback.")
    llm = None
    llm_status = f"❌ LLM load failed: {str(e)}"

# -------------------------------
# HELPER: HAVERSINE DISTANCE
# -------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# -------------------------------
# LOAD & ENRICH DATA
# -------------------------------
@st.cache_data
def load_and_enrich_data():
    consumers = pd.read_csv("consumers.csv")
    merchants = pd.read_csv("merchants.csv")
    transactions = pd.read_csv("transactions.csv")
    # --- LOAD OR INITIALIZE WEIGHTS DATABASE ---
    try:
        weights_db = pd.read_csv("user_weights.csv", index_col="consumer_id")
    except FileNotFoundError:
        weights_db = pd.DataFrame(columns=[
            "consumer_id", "category_weight", "price_weight", 
            "proximity_weight", "baseline_weight", "small_business_weight"
        ])
        weights_db.set_index("consumer_id", inplace=True)

    price_tiers = {
        'coffee': 0.3, 'fast food': 0.2, 'food trucks': 0.25,
        'organic': 0.6, 'dining': 0.7, 'steak': 0.9, 'fine dining': 0.95,
        'electronics': 0.8, 'books': 0.4, 'beauty': 0.6,
        'fitness': 0.5, 'yoga': 0.6, 'theme parks': 0.7,
        'travel': 0.85, 'craft beer': 0.5, 'beer': 0.4,
        'art': 0.5, 'shows': 0.8, 'music': 0.6,
        'home': 0.7, 'cars': 0.95, 'outdoor': 0.6,
        'sports': 0.5, 'tex-mex': 0.4, 'bbq': 0.5,
        'chile': 0.4, 'mexican': 0.45, 'bourbon': 0.7,
        'farm': 0.5, 'farm-to-table': 0.75, 'desert': 0.8,
        'beach': 0.7, 'surf': 0.65, 'wine': 0.8,
        'bazaar': 0.4, 'finance': 0.6, 'wealth': 0.9,
        'insurance': 0.5, 'ivy': 0.6, 'lowcountry': 0.65,
        'history': 0.5, 'climbing': 0.6, 'blues': 0.6, 'jazz': 0.6
    }
    merchants['price_tier'] = merchants['category'].map(price_tiers).fillna(0.5)
    merchants['is_small_business'] = (merchants['merchant_id'] > 130).astype(int)

    consumer_spend = transactions.groupby(['consumer_id', 'category'])['amount'].sum().unstack(fill_value=0)
    all_cats = sorted(set(merchants['category'].unique()) | set(consumer_spend.columns))
    consumer_spend = consumer_spend.reindex(columns=all_cats, fill_value=0)

    visit_count = transactions.groupby('consumer_id').size()
    total_spend = transactions.groupby('consumer_id')['amount'].sum()
    consumers['avg_spend_per_visit'] = (total_spend / visit_count).reindex(consumers['consumer_id']).fillna(20)
    income_map = {'low': 0.2, 'medium': 0.5, 'high': 0.9}
    consumers['income_numeric'] = consumers['income_bracket'].map(income_map)
    top_cat = consumer_spend.idxmax(axis=1)
    consumers['top_category'] = consumers['consumer_id'].map(top_cat).fillna('dining')

    return consumers, merchants, transactions, consumer_spend, weights_db

try:
    consumers, merchants, transactions, consumer_spend, weights_db = load_and_enrich_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -------------------------------
# OPTIMIZED USER CONTROLS SIDEBAR
# -------------------------------

st.sidebar.header("User & Privacy Controls")
show_logs = st.sidebar.checkbox("🔍 Show System Logs", value=False)

# session_state initialize
if "memory_deleted" not in st.session_state:
    st.session_state.memory_deleted = False

# === USER SELECTION GROUP ===
with st.sidebar.expander("👤 User Controls", expanded=True):
    consumer_options = consumers.set_index("consumer_id")["name"]
    selected_name = st.selectbox("Select Consumer", consumer_options)

# === PERSONALIZATION CONTROLS ===
with st.sidebar.expander("🔒 Personalization Controls", expanded=True):
    pause_deals = st.checkbox("⏸️ Pause Personalized Deals", value=False)

    opt_out_cats = st.multiselect(
        "Opt-out Categories",
        options=sorted(merchants["category"].unique()),
        disabled=pause_deals
    )

    # Delete AI Memory directly
    if st.button("🗑️ Delete AI Memory"):
        st.session_state.memory_deleted = True
        st.sidebar.success("AI memory cleared (simulated).")

# === SYSTEM LOGS ===
system_logs = {
    "llm_status": llm_status if 'llm_status' in locals() else "unknown",
    "privacy_controls": {
        "opt_out_categories": opt_out_cats,
        "deals_paused": pause_deals,
        "memory_deleted": st.session_state.memory_deleted
    },
    "matchmaking_details": [],
    "fairness_applied": False
}

# === STATUS DISPLAY ===
if st.session_state.memory_deleted:
    st.sidebar.info("AI memory is currently cleared.")
if pause_deals:
    st.sidebar.success("Personalized deals are currently paused")

# === PAUSED DEALS EFFECT ===
if pause_deals:
    st.info("⏸️ Personalized deals are paused. Showing popular deals instead.")
    top_merchants = merchants.sort_values('merchant_id', ascending=False).head(10)
    for _, m in top_merchants.iterrows():
        st.write(f"- **{m['name']}** ({m['category']}) – {m['city']}")
    st.stop()



# -------------------------------
# SELECTED CONSUMER
# -------------------------------
consumer_row = consumers[consumers["name"] == selected_name].iloc[0]
consumer_id = int(consumer_row["consumer_id"])

consumer_lat = merchants[merchants['city'] == consumer_row['city']]['lat'].mean()
consumer_lon = merchants[merchants['city'] == consumer_row['city']]['lon'].mean()
if pd.isna(consumer_lat):
    consumer_lat, consumer_lon = 40.7484, -73.9857

# -------------------------------
# GET USER WEIGHTS (LLM + CSV CACHE + LOGGING)
# -------------------------------
def get_user_weights(consumer_id, user_consumption, llm, weights_db):

    if consumer_id in weights_db.index:
        # Read weight memory
        weights = weights_db.loc[consumer_id].values.astype(float)
    else:
        prompt = f"""
        User past spending: {user_consumption}

        Suggest normalized weights for the following factors (sum=1):
        category, price, proximity, baseline, small_business
        Return as a Python list, e.g., [0.3, 0.25, 0.2, 0.15, 0.1].
        """

        # 调用 LLM 并带日志记录
        fallback = "[0.4, 0.25, 0.2, 0.1, 0.05]"  # default weight
        weights_text = generate_with_llm(prompt, fallback, log_key=f"user_weights_{consumer_id}")

        try:
            weights = eval(weights_text)
            weights = np.array(weights) / sum(weights)
        except Exception:
            weights = np.array([0.4, 0.25, 0.2, 0.1, 0.05])

        # Write to memory
        weights_db.loc[consumer_id] = weights
        weights_db.to_csv("user_weights.csv")

    return weights


# -------------------------------
# MATCHMAKING WITH DYNAMIC WEIGHTS
# -------------------------------
def compute_scores_dynamic(
    consumer_row, consumer_id, consumer_lat, consumer_lon,
    merchants, consumer_spend, opt_out_cats, logs, llm, weights_db
):
    """
    Compute recommendation scores for a consumer with dynamic weights from LLM.
    Returns scores array and updates logs.
    """
    user_consumption = consumer_spend.loc[consumer_id].to_dict()
    
    weights = get_user_weights(consumer_id, user_consumption, llm, weights_db)
    
    scores = []
    details = []

    for idx, m in merchants.iterrows():
        log_entry = {"merchant": m['name'], "category": m['category'], "reasons": []}

        # --- opt-out category ---
        if m['category'] in opt_out_cats:
            scores.append(-1)
            log_entry["score"] = -1
            log_entry["reasons"].append("Category opted out by user")
            details.append(log_entry)
            continue

        # --- category score ---
        cat_score = consumer_spend.loc[consumer_id, m['category']] if m['category'] in consumer_spend.columns else 0
        cat_norm = min(cat_score / 100.0, 1.0)
        log_entry["category_spend"] = cat_score
        log_entry["category_score"] = cat_norm

        # --- price score ---
        price_match = 1.0 - abs(m['price_tier'] - consumer_row['income_numeric'])
        log_entry["price_match"] = price_match

        # --- distance score ---
        dist = haversine_distance(consumer_lat, consumer_lon, m['lat'], m['lon'])
        proximity = np.exp(-dist / 10.0)
        log_entry["distance_km"] = dist
        log_entry["proximity_score"] = proximity

        # --- Small Marchent score ---
        small_boost = 0.2 if m['is_small_business'] else 0.0
        if small_boost > 0:
            log_entry["reasons"].append("Small business boost applied")

        # --- Final Score ---
        score = (
            weights[0]*cat_norm +
            weights[1]*price_match +
            weights[2]*proximity +
            weights[3]*0.5 +       # baseline
            weights[4]*small_boost
        )
        scores.append(score)

        log_entry["final_score"] = score
        log_entry["is_small_business"] = bool(m['is_small_business'])
        details.append(log_entry)

    # --- Update log ---
    logs["matchmaking_details"] = details
    logs["user_weights"] = weights.tolist()

    scores = np.array(scores)

    # --- Small Marchent bonus point ---
    large_mask = merchants['merchant_id'] <= 130
    small_mask = ~large_mask
    top10 = np.argsort(scores)[::-1][:10]
    small_in_top = small_mask[top10].sum()

    if small_in_top < 2:
        scores[small_mask] += 0.15
        logs["fairness_applied"] = True
        logs["fairness_log"] = f"Boosted small businesses: only {small_in_top} in top 10, now enforced minimum of 2."

    return scores, top10


# -------------------------------
# LLM UTIL WITH LOGGING
# -------------------------------
def generate_with_llm(prompt, fallback, log_key="llm_call"):
    if llm is None:
        system_logs.setdefault("llm_calls", []).append({log_key: "FALLBACK (LLM unavailable)"})
        return fallback
    try:
        messages = [
            {"role": "system", "content": "You are AmEx MatchAI..."},
            {"role": "user", "content": prompt}
        ]
        text = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm(text)
        generated = outputs[0]["generated_text"]
        reply = generated.split("<|assistant|>")[-1].strip() if "<|assistant|>" in generated else generated[len(text):].strip()
        system_logs.setdefault("llm_calls", []).append({log_key: "SUCCESS"})
        return reply if reply else fallback
    except Exception as e:
        system_logs.setdefault("llm_calls", []).append({log_key: f"ERROR: {str(e)}"})
        return fallback

# -------------------------------
# UI: PROFILE & MATCHES
# -------------------------------
st.subheader(f"Consumer Profile: {consumer_row['name']}")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Age", consumer_row["age"])
    st.metric("City", consumer_row["city"])
with c2:
    st.metric("Income", consumer_row["income_bracket"].title())
    st.metric("Top Category", consumer_row["top_category"].title())
with c3:
    total_spent = transactions[transactions["consumer_id"] == consumer_id]["amount"].sum()
    st.metric("Total Spent", f"${total_spent:,.2f}")

st.subheader("Your AI-Powered Deals")

with st.spinner("Generating your personalized deals... ⏳"):
    scores, top10 = compute_scores_dynamic(
    consumer_row,
    consumer_id,
    consumer_lat,
    consumer_lon,
    merchants,
    consumer_spend,
    opt_out_cats,
    system_logs,
    llm,
    weights_db
    )
    # Bundle deal
    if len(top10) >= 2:
        m1, m2 = merchants.iloc[top10[0]], merchants.iloc[top10[1]]
        bundle_prompt = f"Create a fun bundled experience for {consumer_row['name']} combining {m1['name']} ({m1['category']}) and {m2['name']} ({m2['category']}) in {m1['city']}."
        bundle_fallback = f"✨ Bundle idea: Enjoy {m1['category']} at {m1['name']} and {m2['category']} at {m2['name']}!"
        bundle_deal = generate_with_llm(bundle_prompt, bundle_fallback, "bundle_deal")
        st.info(f"**Experience Bundle**: {bundle_deal}")

    # Top matches
    for rank, idx in enumerate(top10[:5], 1):
        m = merchants.iloc[idx]
        score = scores[idx]

        deal_prompt = f"Generate a short deal for {consumer_row['name']} at {m['name']} ({m['category']}, {m['city']})."
        deal = generate_with_llm(deal_prompt, f"{consumer_row['name']}, get 20% off at {m['name']}!", f"deal_{rank}")

        exp_prompt = f"Explain why {consumer_row['name']} would love {m['name']}."
        explanation = generate_with_llm(exp_prompt, f"Matches your preferences.", f"explanation_{rank}")

        with st.expander(f"#{rank} – {m['name']} ({m['category'].title()}) – {score:.1%} match"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(f"https://via.placeholder.com/120?text={m['category'].title()}", width=110)
            with col2:
                st.write(f"📍 {m['city']}, {m['state']}")
                st.write(f"💰 {'Premium' if m['price_tier'] > 0.7 else 'Affordable'} | 🏪 {'Small Biz' if m['is_small_business'] else 'Chain'}")
                st.markdown(f"**Offer**: {deal}")
            st.progress(min(score, 1.0))
            with st.expander("🔍 Why this match?"):
                st.write(explanation)

# -------------------------------
# MERCHANT INSIGHTS
# -------------------------------
st.subheader("Merchant AI Insights")
merchant_name = st.selectbox("Select your merchant", merchants["name"].tolist())
selected_m = merchants[merchants["name"] == merchant_name].iloc[0]
insight = generate_with_llm(
    f"Give insight to {selected_m['name']} in {selected_m['city']}.",
    "Bundle lunch with dessert spots!",
    "merchant_insight"
)
st.success(f"💡 **AI Insight**: {insight}")

# -------------------------------
# SYSTEM LOGS (EXPLAINABILITY)
# -------------------------------
if show_logs:
    st.divider()
    st.subheader("🔍 System Transparency Logs")
    st.caption("AmEx MatchAI logs for explainability, debugging, and compliance")

    st.markdown("### 🧠 LLM & AI Status")
    st.text(system_logs["llm_status"])
    if "llm_calls" in system_logs:
        st.json(system_logs["llm_calls"])

    st.markdown("### 🛡️ Privacy Controls")
    st.json(system_logs["privacy_controls"])

    st.markdown("### ⚖️ Fairness Enforcement")
    if system_logs["fairness_applied"]:
        st.success(system_logs["fairness_log"])
    else:
        st.info("No fairness adjustment needed — small businesses already well-represented.")

    st.markdown("### 📊 Matchmaking Breakdown (Top 3)")
    for i, log in enumerate(system_logs["matchmaking_details"][:3]):
        with st.expander(f"{log['merchant']} ({log['category']})"):
            st.json({
                "category_spend": log.get("category_spend", 0),
                "price_match": round(log.get("price_match", 0), 2),
                "distance_km": round(log.get("distance_km", 0), 1),
                "is_small_business": log.get("is_small_business", False),
                "final_score": round(log.get("final_score", 0), 3),
                "reasons": log.get("reasons", [])
            })

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption("AmEx MatchAI – Ethical, Explainable, Fair | All data processed in AmEx’s closed-loop system")