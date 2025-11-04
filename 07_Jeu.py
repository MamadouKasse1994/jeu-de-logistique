import streamlit as st
import random
import numpy as np
import pandas as pd
import uuid
from copy import deepcopy
import json
import itertools 
import time
import datetime
# --- NOUVEAUX IMPORTS POUR LE MULTIJOUEUR ---
from supabase import create_client, Client
# ---------------------------------------------

# ---------------- CONFIGURATION & PARAMÈTRES GLOBALES ----------------

# --- Constantes de Jeu ---
BASE_DEMANDS = {"A": 450, "B": 350, "C": 250, "D": 150, "E": 100} 
SALARY_PER_EMP = 2500 
FIXED_COSTS = 5000 
INSURANCE_PER_TRUCK_BASE = 600 
TAX_RATE = 0.15 
BASE_PRICES = {"A": 150, "B": 180, "C": 250, "D": 350, "E": 500} 
MIN_TRUCK_RESALE_RATIO = 0.35
PERF_LOSS_PER_AGE = 0.015 
INDEMNITY_PER_EMP = 1800
TOTAL_CLIENTS = 30
COLIS_TYPES = ["A", "B", "C", "D", "E"]
REFERENCE_SPEED = 80 
MAX_TRIPS = 3 
R_D_COST_BASE = 20000 
FAILLITE_RATIO = 0.8 # Si dette > 80% valeur actifs, l'entreprise est en danger
MAX_LOAN_AGE_BEFORE_SEIZURE = 2
MAX_LOAN_CAPACITY_RATIO = 5 
INTEREST_RATE_PER_TURN = 0.03
MIN_LOAN_PAYMENT_RATIO = 0.15
CAPACITY_UNITS_PER_COLIS = {"A": 1.0, "B": 1.5, "C": 2.0, "D": 3.0, "E": 5.0}

TRUCK_MODELS = [
    {"id": "M1 (Lent)", "speed": 80, "capacity": 25, "price": 25000, "maintenance": 500, "purchase_price": 25000, "age": 0},
    {"id": "M2 (Moyen)", "speed": 100, "capacity": 35, "price": 40000, "maintenance": 900, "purchase_price": 40000, "age": 0},
    {"id": "M3 (Rapide)", "speed": 120, "capacity": 45, "price": 60000, "maintenance": 1300, "purchase_price": 60000, "age": 0}
]

R_D_TYPES = {
    "Logistique": {"cost": R_D_COST_BASE, "effect": "Améliore l'efficacité des camions (Capacité +5%).", "boost_value": 0.05},
    "CyberSécurité": {"cost": R_D_COST_BASE * 1.2, "effect": "Protège des cyberattaques et des pannes IT."},
    "Carburant": {"cost": R_D_COST_BASE * 0.9, "effect": "Protège des pénuries et fluctuations des coûts de carburant."}
}

EVENT_LIST = [
    {"name": "Piratage Bancaire Majeur", "type": "Cyber", "text": "Une faille de sécurité majeure affecte la confiance des clients.", "rep_penalty": 0.8, "rep_penalty_uncovered": 0.65},
    {"name": "Grève des Transporteurs", "type": "Logistique", "text": "La demande est réduite de 25%.", "market_effect": 0.75, "market_bonus_covered": 1.10},
    {"name": "Pénurie de Carburant", "type": "Carburant", "text": "Le prix du carburant double. Les coûts fixes et variables augmentent de 50%.", "cost_increase": 1.5, "cost_protection_covered": 1.1},
    {"name": "Nouvelle Route Commerciale", "type": "Market", "text": "Une nouvelle route s'ouvre. Demande accrue de 25%.", "market_effect": 1.25},
    {"name": "Changement de Réglementation", "type": "Reglementation", "text": "Nouvelles normes de sécurité. Frais imprévus plus élevés.", "rep_penalty": 0.95, "unforeseen_cost_mod": 1.5}
]

# ---------------- CONFIGURATION SUPABASE ----------------

@st.cache_resource
def init_supabase():
    """Initialise le client Supabase."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError:
        # Nécessaire pour l'environnement de l'assistant. En réel, on lèverait une erreur.
        return None 

supabase: Client = init_supabase()

# ---------------- FONCTIONS DE SYNCHRONISATION MULTIJOUEUR ----------------

def to_serializable(obj):
    """Gère la conversion d'objets non-JSON-compatibles (uuid, datetime, etc.) en str."""
    if isinstance(obj, uuid.UUID) or isinstance(obj, datetime.datetime):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_game_state_to_db(game_id, game_state):
    """Sauvegarde l'état complet du jeu dans Supabase."""
    if not supabase: return 
    
    game_state_data = deepcopy(game_state)

    keys_to_save = ['game_id', 'turn', 'market_trend', 'backlog_packages', 'event_history',
                    'current_event', 'players', 'num_ia_players', 'host_name',
                    'actions_this_turn', 'players_ready', 'game_ready',
                    'game_status', 'pending_players', 'host_participates']

    state_to_save = {k: game_state_data.get(k) for k in keys_to_save if k in game_state_data}

    data_to_save = {
        "game_id": game_id,
        "state_json": json.dumps(state_to_save, default=to_serializable),
        "turn": game_state_data.get('turn', 1),
        "updated_at": datetime.datetime.now().isoformat()
    }

    try:
        response = supabase.table("games").upsert(data_to_save).execute()
        return response
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde: {e}")
        return None

def load_game_state_from_db(game_id):
    """Loads the complete game state from Supabase, specifically the state_json."""
    if not supabase: return None

    try:
        # CORRECTION: Changement de .single() à .limit(1) pour prévenir l'erreur PGRST116
        # si plusieurs enregistrements ont le même game_id, et prendre le premier trouvé.
        response = supabase.table("games").select("state_json").eq("game_id", game_id).limit(1).execute()
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'état du jeu depuis la base de données: {e}")
        return None

    # response.data est une liste. Nous vérifions si elle n'est pas vide et si le premier élément contient state_json.
    if response.data and len(response.data) > 0 and response.data[0].get("state_json"):
        try:
            # Accéder à state_json à partir du premier élément de la liste
            loaded_state_data = json.loads(response.data[0]["state_json"])
            return loaded_state_data
        except json.JSONDecodeError:
            st.error("Erreur de décodage JSON de l'état du jeu chargé.")
            return None
    else:
        return None

def update_session_from_db(loaded_state_data):
    """Updates st.session_state.game_state from the loaded data."""
    my_name = st.session_state.get('my_name')
    user_name = st.session_state.get('current_user_name')
    game_id = st.session_state.get('game_id') 

    st.session_state.game_state = loaded_state_data

    st.session_state.my_name = my_name
    st.session_state.current_user_name = user_name
    st.session_state.game_id = game_id 

    st.success("État du jeu synchronisé avec succès!")

def sync_game_state(game_id):
    """Forces loading game state from DB and triggers a rerun."""
    loaded_state = load_game_state_from_db(game_id)
    if loaded_state:
        update_session_from_db(loaded_state)
        st.rerun()
    else:
        st.error("Impossible de se synchroniser. Vérifiez l'ID de la partie.")
        
def update_game_chat(game_id, player_name, message):
    """Ajoute un message au chat de la partie."""
    if not supabase: return
    new_message = {
        "game_id": game_id,
        "sender": player_name,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        supabase.table("chat_messages").insert(new_message).execute()
    except Exception as e:
        st.error(f"Erreur lors de l'envoi du message de chat: {e}")

def load_game_chat(game_id):
    """Charge les 10 derniers messages du chat."""
    if not supabase: return []
    try:
        response = supabase.table("chat_messages").select("sender, message, timestamp").eq("game_id", game_id).order("timestamp", desc=True).limit(10).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Erreur lors du chargement du chat: {e}")
        return []


# ---------------- GESTION DE L'ÉTAT DU JEU (Initialisation/Adaptation) ----------------

def _new_truck(model):
    new_truck = deepcopy(model)
    new_truck["uuid"] = uuid.uuid4() 
    return new_truck

def generate_player_names(num_ia, existing_names):
    """Génère de nouveaux noms d'IA uniques."""
    names = []
    used_initials = {name.split(' ')[1] for name in existing_names if len(name.split(' ')) > 1}
    
    ia_letters = itertools.cycle('BCDEFGHIJKLMNOPQRSTUVWXYZ') 
    
    for i in range(num_ia):
        new_name = ""
        while True:
            letter = next(ia_letters)
            if letter == 'A': continue 
            new_name = f"Ent. {letter} (IA)"
            if new_name not in existing_names and letter not in used_initials:
                used_initials.add(letter)
                break
        names.append(new_name)
    return names

def create_player_entity(name, is_human):
    """Crée l'objet d'entité joueur."""
    return {
        "name": name,
        "is_human": is_human,
        "money": 50000,
        "loan": 0,
        "loan_age": 0,
        "reputation": 1.0,
        "employees": 5,
        "trucks": [_new_truck(TRUCK_MODELS[0]) for _ in range(2)],
        "prices": deepcopy(BASE_PRICES),
        "active": True,
        "can_recover": True,
        "rd_boost_log": 0,
        "rd_investment_type": "Aucun",
        "history": ["Initialisation du jeu."],
        "delivered_packages_total": {t: 0 for t in COLIS_TYPES},
        "income": 0, "expenses": 0, "asset_value": 0, "total_capacity": 0,
    }

def initialize_game_state(host_player_name, num_ia_players, host_participates):
    """Crée l'état initial du jeu avec l'hôte et les IAs dans une structure imbriquée."""
    game_id = f"GAME-{uuid.uuid4().hex[:6].upper()}"

    game_state_data = {
        "game_id": game_id,
        "turn": 1,
        "market_trend": 1.0,
        "backlog_packages": {t: 0 for t in COLIS_TYPES},
        "event_history": [],
        "current_event": {"name": "Initialisation", "text": "Le jeu commence.", "type": "None"},
        "players": [],
        "num_ia_players": num_ia_players, 
        "host_name": host_player_name,
        "host_participates": host_participates,
        "actions_this_turn": {},
        "players_ready": {},
        "game_ready": True,
        "game_status": "lobby",
        "pending_players": []
    }
    
    existing_names = []

    # 1. Ajout de l'hôte
    if host_participates:
        host_entity_name = host_player_name 
        host_entity = create_player_entity(host_entity_name, True)
        game_state_data["players"].append(host_entity)
        game_state_data["players_ready"][host_player_name] = False
        existing_names.append(host_entity_name)
    else:
        host_ia_name = f"{host_player_name} (IA Host)"
        host_entity = create_player_entity(host_ia_name, False)
        game_state_data["players"].append(host_entity)
        existing_names.append(host_ia_name)

    # 2. Ajout des IAs concurrentes initiales
    initial_ia_names = generate_player_names(num_ia_players, existing_names)
    for name in initial_ia_names:
        game_state_data["players"].append(create_player_entity(name, False))
        existing_names.append(name)


    st.session_state.game_state = game_state_data
    st.session_state.my_name = host_player_name
    st.session_state.game_id = game_id

    save_game_state_to_db(game_id, st.session_state.game_state)

    return st.session_state.game_state


def show_lobby_host(game_id, host_name):
    """Interface du lobby pour le contrôleur (Host)."""

    st.title("Admin de la Partie et Lobby")
    st.info(f"ID de la Partie: **{game_id}** | Contrôleur: **{host_name}**")

    st.markdown("---")

    # === BLOC DE SYNCHRONISATION/ACTUALISATION DÉDIÉ ===
    st.subheader("🔁 Synchronisation du Lobby")

    if st.button("Actualiser le Lobby (Voir les Nouveaux Joueurs)", type="secondary", use_container_width=True):
        st.info("Synchronisation forcée...")
        sync_game_state(game_id)

    st.caption("Pour une meilleure expérience, demandez aux joueurs invités de cliquer sur le bouton 'Actualiser le Statut du Jeu' dans la barre latérale.")
    st.markdown("---")
    # =================================================

    # 1. Gestion des joueurs en attente
    st.subheader("🚪 Joueurs en Attente d'Approbation")

    pending_players = st.session_state.game_state.get('pending_players', [])

    if pending_players:
        for player_name in pending_players:
            col_name, col_accept, col_reject = st.columns([3, 1, 1])
            col_name.write(f"**{player_name}**")

            if col_accept.button("✅ Accepter", key=f"accept_{player_name}"):
                new_player_entity = create_player_entity(player_name, True)
                st.session_state.game_state["players"].append(new_player_entity)
                st.session_state.game_state["players_ready"][player_name] = False
                st.session_state.game_state["pending_players"].remove(player_name)

                save_game_state_to_db(game_id, st.session_state.game_state)
                st.success(f"Joueur {player_name} accepté. Vous devez lancer le jeu.")
                st.rerun()

            if col_reject.button("❌ Rejeter", key=f"reject_{player_name}"):
                st.session_state.game_state["pending_players"].remove(player_name)
                save_game_state_to_db(game_id, st.session_state.game_state)
                st.warning(f"Joueur {player_name} rejeté.")
                st.rerun()
    else:
        st.info("Aucun joueur en attente pour l'instant.")

    st.markdown("---")

    # 2. Liste des joueurs acceptés et IA
    st.subheader("👥 Joueurs Participants (Humains et IA)")
    all_players = st.session_state.game_state.get('players', [])
    
    host_entity = next((p for p in all_players if p["name"] == host_name and p["is_human"]), None)
    host_ia_entity = next((p for p in all_players if p["name"] == f"{host_name} (IA Host)" and not p["is_human"]), None)
    
    def get_role(player_data):
        if player_data == host_entity:
            return "Contrôleur/Joueur"
        elif player_data == host_ia_entity:
            return "Contrôleur (IA Host)"
        elif player_data["is_human"]:
            return "Joueur"
        else:
            return "IA"

    if all_players:
        df_players = pd.DataFrame([
            {"Nom": p['name'], "Type": "Humain" if p['is_human'] else "IA", "Rôle": get_role(p)}
            for p in all_players
        ])
        st.dataframe(df_players, hide_index=True, use_container_width=True)
    else:
        st.warning("Aucun joueur dans la partie.")

    st.markdown("---")

    # 3. Ajouter des joueurs IA supplémentaires
    st.subheader("➕ Ajouter des Joueurs IA")
    num_new_ia = st.number_input("Nombre d'IA supplémentaires à ajouter", min_value=0, max_value=10, value=0, key="add_ia_number")

    if st.button("Ajouter ces IA"):
        if num_new_ia > 0:
            existing_names = [p['name'] for p in st.session_state.game_state["players"]]
            new_ia_names = generate_player_names(num_new_ia, existing_names)

            for name in new_ia_names:
                st.session_state.game_state["players"].append(create_player_entity(name, False))
                st.session_state.game_state["num_ia_players"] += 1 

            save_game_state_to_db(game_id, st.session_state.game_state)
            st.success(f"{num_new_ia} IA(s) ajoutée(s) avec succès.")
            st.rerun()
        else:
            st.info("Entrez un nombre d'IA supérieur à 0 pour ajouter.")

    st.markdown("---")

    # 4. Lancement de la partie
    st.subheader("Lancer la Partie")
    human_players_exist = any(p['is_human'] for p in st.session_state.game_state.get('players', []))
    disable_start = not human_players_exist 

    if st.button("▶️ Lancer la Partie Maintenant", type="primary", disabled=disable_start):
        st.session_state.game_state['game_status'] = 'in_progress'
        human_players_entities = [p for p in st.session_state.game_state["players"] if p['is_human']]
        st.session_state.game_state["players_ready"] = {p["name"]: False for p in human_players_entities}

        save_game_state_to_db(game_id, st.session_state.game_state)
        st.rerun()

    if not human_players_exist:
        st.warning("Le lancement de la partie est désactivé car il n'y a aucun joueur humain.")
    st.caption("Une fois lancé, de nouveaux joueurs ne pourront plus rejoindre.")


# ---------------- FONCTIONS DE CALCUL ----------------

def calculate_player_capacity(player_data):
    total_capacity = 0
    log_rd_boost = player_data.get("rd_boost_log", 0)
    
    for truck in player_data["trucks"]:
        if not isinstance(truck, dict) or 'id' not in truck: continue 
        
        perf_factor = 1.0 - (truck["age"] * PERF_LOSS_PER_AGE)
        perf_factor = max(0.6, perf_factor) 
        effective_speed = truck["speed"] * perf_factor
        effective_capacity = truck["capacity"] * perf_factor * (1 + log_rd_boost)
        trip_multiplier = min(MAX_TRIPS, effective_speed / REFERENCE_SPEED) 
        total_capacity += int(effective_capacity * trip_multiplier)
        
    return total_capacity

def calculate_asset_value(player_trucks):
    total_value = 0
    for truck in player_trucks:
        if not isinstance(truck, dict) or 'id' not in truck: continue
        
        current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
        resale = max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value)
        total_value += resale
    return int(total_value)

def poisson_market(base, trend=1.0):
    return int(np.random.poisson(max(0, base * trend)))

def generate_client_orders(game_state):
    package_orders = {t: 0 for t in COLIS_TYPES}
    
    for t in COLIS_TYPES:
        package_orders[t] += game_state["backlog_packages"].get(t, 0)
    
    for _ in range(TOTAL_CLIENTS * 2): 
        types_chosen = random.choices(COLIS_TYPES, k=random.randint(1, 3))
        for t in types_chosen:
            package_orders[t] += random.randint(1, 5)

    for t in COLIS_TYPES:
        package_orders[t] += poisson_market(BASE_DEMANDS.get(t, 0), game_state["market_trend"]) 
        
    if "market_effect" in game_state["current_event"]:
        for t in package_orders:
            package_orders[t] = int(package_orders[t] * game_state["current_event"]["market_effect"])

    capacity_required = {
        t: int(package_orders[t] * CAPACITY_UNITS_PER_COLIS.get(t, 1.0)) 
        for t in COLIS_TYPES
    }
    
    return capacity_required

def calculate_competition_score(p, t):
    player_exec_capacity = p["total_capacity"]
    price_score = p["prices"].get(t, BASE_PRICES.get(t, 500)) * 0.4
    rep_score = 800 / max(1, p["reputation"])
    cap_factor = 1000 / (player_exec_capacity + 1)
    
    total_score = price_score + rep_score + cap_factor
    attractiveness_weight = 1.0 / (total_score * total_score) 
    
    return attractiveness_weight, player_exec_capacity

def distribute_clients(market_capacity_demand, players, game_state):
    allocation_capacity = {p["name"]: {t: 0 for t in COLIS_TYPES} for p in players}
    current_package_backlog = {t: 0 for t in COLIS_TYPES}
    active_players = [p for p in players if p["active"]]
    
    player_data = {}
    for p in active_players:
        p_cap = calculate_player_capacity(p)
        p["total_capacity"] = p_cap
        player_data[p["name"]] = {
            "player": p,
            "max_capacity": p_cap,
            "current_allocation_total": 0,
            "scores": {t: calculate_competition_score(p, t)[0] for t in COLIS_TYPES}
        }
    
    for t in COLIS_TYPES:
        qty_capacity_remaining = market_capacity_demand.get(t, 0)
        colis_size = CAPACITY_UNITS_PER_COLIS.get(t, 1.0)
        
        if qty_capacity_remaining == 0: continue
        unit_size = max(1, qty_capacity_remaining // 4) 
        
        while qty_capacity_remaining > 0:
            scores_and_weights = []
            for p_name, data in player_data.items():
                p = data["player"]
                cap_remaining_global = data["max_capacity"] - data["current_allocation_total"]
                
                if cap_remaining_global > 0:
                    scores_and_weights.append({
                        "player": p, 
                        "weight": data["scores"].get(t, 0),
                        "cap_remaining_global": cap_remaining_global
                    })

            total_market_weight = sum(item["weight"] for item in scores_and_weights)
            
            if not scores_and_weights or total_market_weight == 0:
                break
            
            weights = [item["weight"] for item in scores_and_weights]
            chosen_items = random.choices(scores_and_weights, weights=weights, k=1)
            
            if chosen_items:
                chosen_item = chosen_items[0]
                p = chosen_item["player"]
                cap_remaining = chosen_item["cap_remaining_global"]
                p_name = p["name"]
                
                capacity_to_distribute = min(unit_size, qty_capacity_remaining) 
                deliverable_capacity = min(capacity_to_distribute, cap_remaining)
                
                if deliverable_capacity > 0:
                    allocation_capacity[p_name][t] += deliverable_capacity
                    qty_capacity_remaining -= deliverable_capacity
                    player_data[p_name]["current_allocation_total"] += deliverable_capacity
                else:
                    pass
            else:
                break
                
        capacity_unallocated = max(0, qty_capacity_remaining)
        packages_unallocated = int(capacity_unallocated / colis_size)
        current_package_backlog[t] += packages_unallocated
        
    for t in current_package_backlog:
        game_state["backlog_packages"][t] = min(20, current_package_backlog[t])
            
    return allocation_capacity

def trigger_random_event(game_state):
    if random.random() < 0.4: 
        event = random.choice(EVENT_LIST)
        game_state["current_event"] = event
        game_state["event_history"].append(f"Tour {game_state['turn']}: {event['name']} - {event['text']}")
    else:
        game_state["current_event"] = {"name": "Aucun", "text": "Un tour normal.", "type": "None"}
        
# ---------------- LOGIQUE D'IA ----------------

def get_ia_actions(player_data):
    """Détermine les actions d'un joueur IA basé sur une stratégie simple."""
    actions = {}
    
    new_prices = deepcopy(player_data["prices"])
    
    # 1. Gestion de la faillite (Vente d'urgence)
    if not player_data["active"]:
        # Pour une IA, on vend un seul camion pour tenter de récupérer
        actions["sell_trucks"] = {}
        trucks_sorted = sorted(player_data["trucks"], key=lambda t: t["age"] * t["price"], reverse=True)
        if trucks_sorted:
            model_id = trucks_sorted[0]["id"]
            # En IA, on vend juste le premier camion trouvé de ce modèle
            truck_uuid = str(trucks_sorted[0]["uuid"])
            actions["sell_trucks"][model_id] = [truck_uuid]
        return actions 

    # 2. Gestion des prix
    price_mod = 0.98 if player_data["reputation"] < 1.0 else 1.02
    for t in COLIS_TYPES:
        if t in ["A", "B"]:
            new_prices[t] = int(BASE_PRICES[t] * price_mod * (1 - (player_data["rd_boost_log"] / 2)))
        else:
            new_prices[t] = int(BASE_PRICES[t] * price_mod)
    actions["prices"] = new_prices

    # 3. Investissement et Capacité
    money_threshold = 40000 + (player_data["turn"] * 5000)
    current_capacity = player_data["total_capacity"]
    
    # Priorité R&D Logistique
    if player_data["money"] > money_threshold and player_data["rd_boost_log"] < 0.2:
        actions["rd_type"] = "Logistique"
    elif player_data["money"] > money_threshold * 1.5 and player_data["rd_investment_type"] == "Aucun":
        actions["rd_type"] = random.choice(["Carburant", "CyberSécurité"])
    else:
        actions["rd_type"] = "Aucun"
        
    # Achat de camions
    actions["buy_trucks"] = {}
    if current_capacity < 1500 and player_data["money"] > 60000:
        model_to_buy = random.choice([TRUCK_MODELS[0], TRUCK_MODELS[1]])
        actions["buy_trucks"][model_to_buy["id"]] = 1
    elif current_capacity < 3000 and player_data["money"] > 100000:
        model_to_buy = TRUCK_MODELS[2]
        actions["buy_trucks"][model_to_buy["id"]] = 1

    # 4. Gestion des employés
    target_employees = max(5, int(current_capacity / 1000))
    emp_delta = target_employees - player_data["employees"]
    if abs(emp_delta) > 1:
        actions["emp_delta"] = 1 if emp_delta > 0 else -1
    
    # 5. Publicité (occasionnelle)
    if player_data["reputation"] < 1.5 and player_data["money"] > 25000 and random.random() < 0.2:
        actions["pub_type"] = "Nationale"

    return actions

# ---------------- LOGIQUE DE JEU INTÉGRÉE (SIMULATE_TURN) ----------------

def simulate_turn_streamlit(game_state, actions_dict):
    """
    Exécute un tour de simulation en utilisant l'état stocké dans game_state
    et les actions des joueurs passées dans actions_dict.
    """
    
    # --- PHASE PRÉ-TOUR ---
    trigger_random_event(game_state)
    current_event = game_state["current_event"]
    event_info = f"🌪️ Événement du Tour: **{current_event['name']}** - {current_event['text']}"
    
    # 1. Actions IA (Calcule la capacité ici pour l'IA avant les actions)
    for i, p in enumerate(game_state["players"]):
        if not p["is_human"]:
            p_cap = calculate_player_capacity(p)
            p["total_capacity"] = p_cap
            ia_action = get_ia_actions(p)
            actions_dict[p["name"]] = ia_action
    
    market_capacity_demand = generate_client_orders(game_state) 
    
    # --- DISTRIBUTION DES COLIS ---
    # La capacité est recalculée pour tous les joueurs dans distribute_clients
    allocation_capacity = distribute_clients(market_capacity_demand, game_state["players"], game_state)

    # --- PHASE D'APPLICATION DES ACTIONS DÉCIDÉES PAR LES JOUEURS ---
    for i, p in enumerate(game_state["players"]):
        
        p["history"] = [event_info]
        action = actions_dict.get(p["name"], {"prices": p["prices"]}).copy()
        
        p["prices"] = action.get("prices", p["prices"])
        p["rd_boost_log"] = p.get("rd_boost_log", 0) 
        p["rd_investment_type"] = action.get("rd_type", "Aucun")
        p["asset_value"] = calculate_asset_value(p["trucks"])

        # 0. Gestion des faillites (Vente d'actifs pour récupérer)
        if not p["active"] and not p.get("can_recover", True):
            p["history"].append("🚨 Entreprise liquidée. Aucune action possible.")
            continue
            
        if not p["active"] and p.get("can_recover", True):
            if "sell_trucks" in action:
                # Dans ce cas, 'sell_trucks' est une liste d'UUIDs à vendre (humain ou IA)
                for model_id, uuid_list in action["sell_trucks"].items():
                    for truck_uuid_str in uuid_list:
                        truck_to_sell = next((t for t in p["trucks"] if str(t.get("uuid")) == truck_uuid_str), None)
                        if truck_to_sell:
                            p["trucks"].remove(truck_to_sell)
                            current_value = truck_to_sell["purchase_price"] * (1 - truck_to_sell["age"] * 0.10) 
                            resale = int(max(truck_to_sell["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                            p["money"] += resale
                            p["history"].append(f"Vente (Faillite): {truck_to_sell['id']} (+{resale:,} €)".replace(",", " "))
                        
            if p["money"] >= 0:
                asset_val = calculate_asset_value(p["trucks"])
                if p["loan"] / max(1, asset_val) < FAILLITE_RATIO:
                    p["active"] = True
                    p["history"].append("Sortie de Faillite! Solde positif et dette sous contrôle.")
            
            if not p["active"]: 
                p["history"].append("Faillite temporaire: doit vendre plus ou rembourser dette.")
                game_state["players"][i] = p
                continue 
        
        # A. Prêts bancaires
        loan_amount = action.get("loan_request", 0)
        loan_payment = action.get("loan_payment", 0)
        
        if loan_amount > 0:
            asset_value = calculate_asset_value(p["trucks"])
            max_loan = asset_value * MAX_LOAN_CAPACITY_RATIO
            if p["loan"] + loan_amount <= max_loan:
                p["money"] += loan_amount
                p["loan"] += loan_amount
                p["loan_age"] = 0 
                p["history"].append(f"Prêt accordé : +{loan_amount:,} €".replace(",", " "))
            else:
                p["history"].append(f"Prêt refusé : Capacité max ({max_loan:,} €) dépassée.".replace(",", " "))
        
        if loan_payment > 0:
            payable = min(loan_payment, p["loan"])
            if p["money"] >= payable:
                p["money"] -= payable
                p["loan"] -= payable
                p["history"].append(f"Remboursement de prêt : -{payable:,} €".replace(",", " "))
            else:
                p["history"].append(f"Remboursement refusé : Fonds insuffisants.".replace(",", " "))


        # B. Recherche & Développement
        rd_type_chosen = action.get("rd_type", "Aucun")
        if rd_type_chosen != "Aucun":
            rd_config = R_D_TYPES.get(rd_type_chosen, {})
            rd_cost = rd_config.get("cost", 0)
            
            if p["money"] >= rd_cost:
                p["money"] -= rd_cost
                p["rd_investment_type"] = rd_type_chosen 
                
                if rd_type_chosen == "Logistique":
                    p["rd_boost_log"] += rd_config.get("boost_value", 0)
                    p["history"].append(f"R&D Logistique : Capacité effective +{rd_config.get('boost_value', 0)*100:.0f}% !".replace(",", " "))
                else:
                    p["history"].append(f"R&D Risque ({rd_type_chosen}) : Couverture activée.")
            else:
                p["rd_investment_type"] = "Aucun"
                p["history"].append(f"R&D ({rd_type_chosen}) refusée: fonds insuffisants.")
        
        # C. Achat/Vente de Camions
        if "buy_trucks" in action:
            for model_id, qty in action["buy_trucks"].items():
                if qty > 0:
                    model = next(m for m in TRUCK_MODELS if m["id"] == model_id)
                    cost = model["price"] * qty
                    if p["money"] >= cost:
                        p["money"] -= cost
                        for _ in range(qty):
                            p["trucks"].append(_new_truck(model)) 
                        p["history"].append(f"Achat : {qty}x {model_id} (-{cost:,} €)".replace(",", " "))
                    else:
                        p["history"].append(f"Achat refusé : Fonds insuffisants pour {model_id}.")
        
        if "sell_trucks" in action and p["active"]: # Ventes volontaires
            for model_id, uuid_list in action["sell_trucks"].items():
                if isinstance(uuid_list, list):
                    for truck_uuid_str in uuid_list:
                        truck_to_sell = next((t for t in p["trucks"] if str(t.get("uuid")) == truck_uuid_str), None)
                        if truck_to_sell:
                            p["trucks"].remove(truck_to_sell)
                            current_value = truck_to_sell["purchase_price"] * (1 - truck_to_sell["age"] * 0.10) 
                            resale = int(max(truck_to_sell["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                            p["money"] += resale
                            p["history"].append(f"Vente volontaire: {truck_to_sell['id']} (+{resale:,} €)".replace(",", " "))
                        else:
                            p["history"].append(f"Vente refusée: Camion {model_id} non trouvé/déjà vendu.")

        # D. Employés et Publicité
        if "emp_delta" in action:
            delta = action["emp_delta"]
            if delta > 0:
                p["employees"] += delta
                p["history"].append(f"Recrutement : +{delta} employé(s).")
            elif delta < 0 and p["employees"] + delta >= 0:
                severance_pay = abs(delta) * INDEMNITY_PER_EMP
                if p["money"] >= severance_pay:
                    p["money"] -= severance_pay
                    p["employees"] += delta
                    p["history"].append(f"Licenciement : {abs(delta)} employé(s) et indemnité de -{severance_pay:,} €.".replace(",", " "))
                else:
                    p["history"].append("Licenciement refusé : Fonds insuffisants pour les indemnités.")
        
        if action.get("pub_type") == "Nationale":
            pub_cost = 15000
            if p["money"] >= pub_cost:
                p["money"] -= pub_cost
                p["reputation"] = min(2.0, p["reputation"] * 1.15) 
                p["history"].append(f"Publicité Nationale : Réputation améliorée (+15%) pour -{pub_cost:,} €.".replace(",", " "))
            else:
                p["history"].append("Publicité refusée : Fonds insuffisants.")


        # E. CALCUL DES RÉSULTATS FINANCIERS DU TOUR
        
        p["income"] = 0
        p["expenses"] = 0

        # 1. Revenus (Income)
        delivered_packages_capacity = allocation_capacity.get(p["name"], {t: 0 for t in COLIS_TYPES})
        revenue = 0
        
        for t in COLIS_TYPES:
            units_delivered = int(delivered_packages_capacity[t] / CAPACITY_UNITS_PER_COLIS.get(t, 1.0))
            price = p["prices"].get(t, BASE_PRICES[t])
            revenue += units_delivered * price
            p["delivered_packages_total"][t] += units_delivered

        p["income"] = int(revenue * (1 - TAX_RATE)) 
        p["money"] += p["income"]
        p["history"].append(f"💰 REVENU NET (Livraisons après Taxes): +{p['income']:,} €".replace(",", " "))

        # 2. Dépenses (Expenses)
        
        # a. Coûts fixes et salaires
        base_expenses = FIXED_COSTS + (p["employees"] * SALARY_PER_EMP) 
        maintenance_costs = sum(truck["maintenance"] for truck in p["trucks"])
        insurance_costs = len(p["trucks"]) * INSURANCE_PER_TRUCK_BASE
        
        # b. Coûts liés à l'événement (avec protection R&D)
        cost_modifier = 1.0
        if current_event["type"] != "None":
            
            if current_event["type"] == "Carburant" and p["rd_investment_type"] == "Carburant":
                cost_modifier = current_event.get("cost_protection_covered", 1.0) 
                p["history"].append("✨ R&D Carburant : Coûts de pénurie réduits (10% d'augmentation seulement).")
            
            elif "cost_increase" in current_event:
                cost_modifier = current_event["cost_increase"] 
                p["history"].append("⚠️ Impact Événement : Coûts fixes et variables augmentés.")
            
        total_variable_costs = int((base_expenses + maintenance_costs + insurance_costs) * cost_modifier)
        
        # c. Frais de prêt et Saisie (Loan Interest and Seizure)
        interest_cost = int(p["loan"] * INTEREST_RATE_PER_TURN)
        p["loan"] += interest_cost 
        p["loan_age"] += 1
        p["history"].append(f"Intérêts de Prêt : +{interest_cost:,} € au principal.".replace(",", " "))
        
        if p["loan_age"] > MAX_LOAN_AGE_BEFORE_SEIZURE and p["loan"] > 0 and not p["active"]:
            trucks_sorted = sorted(p["trucks"], key=lambda t: t["purchase_price"], reverse=True)
            if trucks_sorted:
                seized_truck = trucks_sorted[0]
                p["trucks"].remove(seized_truck)
                seizure_value = int(calculate_asset_value([seized_truck]) * 0.5) 
                p["loan"] = max(0, p["loan"] - seizure_value)
                p["history"].append(f"❌ SAISIE BANCAIRE : Camion {seized_truck['id']} saisi. Dette réduite de {seizure_value:,} €.".replace(",", " "))
                p["loan_age"] = 0 

        # d. Application des dépenses totales
        p["expenses"] = total_variable_costs
        p["money"] -= p["expenses"]
        p["history"].append(f"💸 DÉPENSES TOTALES (Salaires, Maintenance, etc.) : -{p['expenses']:,} €".replace(",", " "))


        # F. ÉVALUATION DE FIN DE TOUR (Réputation, Faillite, Vieillissement)
        
        # 1. Vieillissement des camions
        trucks_to_keep = []
        for truck in p["trucks"]:
            truck["age"] += 1
            if truck["age"] > 10:
                p["history"].append(f"Retrait : Camion {truck['id']} mis au rebut (Trop ancien).")
            else:
                trucks_to_keep.append(truck)
        p["trucks"] = trucks_to_keep
        
        # 2. Mise à jour de la Réputation
        backlog_impact = sum(game_state["backlog_packages"].values()) 
        
        if backlog_impact > 500:
            rep_reduction = 1.0 - (backlog_impact / 20000)
            p["reputation"] = max(0.5, p["reputation"] * rep_reduction)
        
        if "rep_penalty" in current_event:
            penalty = current_event["rep_penalty"]
            
            if current_event["type"] != p["rd_investment_type"]:
                p["reputation"] = max(0.5, p["reputation"] * current_event.get("rep_penalty_uncovered", penalty))
                p["history"].append(f"❌ Malus Réputation : Non-couverture de l'événement {current_event['name']}.")
            else:
                p["reputation"] = min(2.0, p["reputation"] * 1.05)
                p["history"].append(f"✅ Protection R&D : Réputation sauvée de {current_event['name']}.")

        # 3. Vérification de Faillite (nouvelle vérification post-dépenses)
        p["asset_value"] = calculate_asset_value(p["trucks"])
        
        if p["loan"] > 0 and p["loan"] / max(1, p["asset_value"]) >= FAILLITE_RATIO and p["money"] < 0:
            if p["active"]:
                p["active"] = False
                p["history"].append(f"🔴 FAILLITE ! Dette ({p['loan']:,} €) > {FAILLITE_RATIO*100:.0f}% des actifs ({p['asset_value']:,} €). Vendez des actifs ou perdez la partie.".replace(",", " "))
        elif not p["trucks"] and p["money"] < 0:
            p["active"] = False
            p["can_recover"] = False
            p["history"].append(f"☠️ Entreprise liquidée. Tous les actifs sont perdus. GAME OVER.")
        
        # 4. Mise à jour de l'objet joueur
        game_state["players"][i] = p

    # --- PHASE POST-TOUR GLOBAL ---
    game_state["turn"] += 1
    game_state["market_trend"] = max(0.5, min(1.5, game_state["market_trend"] * random.uniform(0.95, 1.05)))
    
    human_players_names = [p["name"] for p in game_state["players"] if p["is_human"]]
    game_state["players_ready"] = {name: False for name in human_players_names}
    
    # Réinitialiser les investissements R&D risque (ils ne durent qu'un tour)
    for p in game_state["players"]:
        if p.get("rd_investment_type", "Aucun") in ["CyberSécurité", "Carburant"]:
             p["rd_investment_type"] = "Aucun"

    return game_state

# ---------------- VUES DE JEU HUMAIN ----------------

def show_game_interface_human(player_name):
    """Affiche le panneau de contrôle pour le joueur humain."""
    
    # Trouver l'entité du joueur actuel
    p = next((p for p in st.session_state.game_state["players"] if p["name"] == player_name and p["is_human"]), None)
    if not p:
        st.error(f"Joueur {player_name} non trouvé dans l'état du jeu. Synchronisation nécessaire.")
        return

    # Si le joueur est prêt, afficher l'écran d'attente
    if st.session_state.game_state["players_ready"].get(player_name, False):
        st.info("✅ **PRÊT POUR LE PROCHAIN TOUR !** En attente des autres joueurs.")
        st.subheader("Messages Récentes du Chat")
        for chat_msg in load_game_chat(st.session_state.game_id):
            st.caption(f"**[{chat_msg['timestamp']}] {chat_msg['sender']}**: {chat_msg['message']}")
        return

    # --- EN TÊTE ET STATUT ---
    st.title(f"Tableau de Bord : {player_name}")
    st.markdown(f"**Tour {st.session_state.game_state['turn']}** | Statut: {'🟢 ACTIF' if p['active'] else '🔴 FAILLITE - Vente d\'actifs urgente'}")

    col_money, col_loan, col_rep, col_cap = st.columns(4)
    col_money.metric("Trésorerie", f"{p['money']:,} €".replace(",", " "))
    col_loan.metric("Dette Bancaire", f"{p['loan']:,} €".replace(",", " "))
    col_rep.metric("Réputation", f"{p['reputation']:.2f}")
    col_cap.metric("Capacité Effective", f"{p['total_capacity']:,} U".replace(",", " "))
    
    st.markdown("---")
    
    # --- FORMULAIRE D'ACTIONS ---
    with st.form("action_form"):
        st.subheader("1. Fixer les Prix des Colis")
        new_prices = {}
        for t in COLIS_TYPES:
            new_prices[t] = st.number_input(f"Prix Colis {t} (Base: {BASE_PRICES[t]} €)", min_value=1, value=p["prices"].get(t, BASE_PRICES[t]), key=f"price_{t}")
        
        st.markdown("---")

        st.subheader("2. Gestion des Actifs et des Employés")
        col_truck, col_emp, col_rd = st.columns(3)
        
        # Achat de Camions
        with col_truck:
            st.markdown("**Achat de Camions**")
            buy_trucks = {}
            for model in TRUCK_MODELS:
                qty = st.number_input(f"Acheter {model['id']} ({model['price']:,} €)".replace(",", " "), min_value=0, max_value=5, value=0, key=f"buy_{model['id']}")
                if qty > 0:
                    buy_trucks[model["id"]] = qty
            st.caption(f"Actifs totaux : {p['asset_value']:,} €".replace(",", " "))

        # Gestion des Employés
        with col_emp:
            st.markdown("**Employés et Pub**")
            emp_delta = st.number_input("Changement d'Employés (Lic. = -1, Recr. = +1)", min_value=-1, max_value=1, value=0, key="emp_delta")
            pub_type = st.radio("Publicité", ["Aucune", "Nationale"], index=0, key="pub_type")
            st.caption(f"Employés actuels : {p['employees']}")
            
        # R&D
        with col_rd:
            st.markdown("**R&D et Protection**")
            rd_type = st.selectbox("Investissement R&D (Un tour)", 
                                  ["Aucun"] + list(R_D_TYPES.keys()), 
                                  key="rd_type_select",
                                  format_func=lambda x: f"{x} (-{R_D_TYPES[x]['cost']:,} €)".replace(",", " ") if x != "Aucun" else "Aucun")
            st.caption("Logistique est permanent. Autres durent 1 tour.")

        st.markdown("---")

        st.subheader("3. Gestion de Trésorerie et Prêts")
        col_loan_req, col_loan_pay, col_sell = st.columns(3)
        
        with col_loan_req:
            loan_request = st.number_input("Demande de Prêt (0 pour ignorer)", min_value=0, step=10000, value=0, key="loan_request")
            st.caption(f"Max empruntable: {int(p['asset_value'] * MAX_LOAN_CAPACITY_RATIO):,} €".replace(",", " "))

        with col_loan_pay:
            loan_payment = st.number_input("Remboursement de Prêt (0 pour ignorer)", min_value=0, step=10000, value=0, key="loan_payment")
            
        with col_sell:
            # Interface de Vente Volontaire (avec sélection par UUID)
            st.markdown("**Vendre Camion (UUID)**")
            sell_trucks_selection = {}
            available_trucks = [{"id": t["id"], "uuid": str(t["uuid"])} for t in p["trucks"]]
            
            if available_trucks:
                truck_options = {t["uuid"]: f"{t['id']} (UUID: {t['uuid'][:4]}...)" for t in available_trucks}
                selected_uuids = st.multiselect("Sélectionner les camions à vendre", 
                                                options=list(truck_options.keys()), 
                                                format_func=lambda x: truck_options.get(x, x),
                                                key="sell_trucks_multiselect")
                
                # Regrouper les UUIDs sélectionnés par modèle ID pour la fonction simulate
                for uuid_str in selected_uuids:
                    truck_model_id = next(t['id'] for t in available_trucks if t['uuid'] == uuid_str)
                    if truck_model_id not in sell_trucks_selection:
                        sell_trucks_selection[truck_model_id] = []
                    sell_trucks_selection[truck_model_id].append(uuid_str)
            else:
                st.info("Aucun camion à vendre.")


        st.markdown("---")
        
        # --- SOUMISSION ---
        submitted = st.form_submit_button("🔒 Verrouiller les Actions pour ce Tour", type="primary", disabled=not p["active"])

    # --- SOUMISSION LOGIC ---
    if submitted:
        
        # Consolider toutes les actions du joueur
        player_actions = {
            "prices": new_prices,
            "buy_trucks": buy_trucks,
            "sell_trucks": sell_trucks_selection,
            "emp_delta": emp_delta,
            "pub_type": pub_type if pub_type != "Aucune" else None,
            "rd_type": rd_type if rd_type != "Aucun" else "Aucun",
            "loan_request": loan_request,
            "loan_payment": loan_payment
        }

        # Enregistrer les actions dans l'état du jeu et marquer le joueur comme prêt
        st.session_state.game_state["actions_this_turn"][player_name] = player_actions
        st.session_state.game_state["players_ready"][player_name] = True
        
        # Sauvegarder et relancer pour afficher l'écran d'attente
        save_game_state_to_db(st.session_state.game_id, st.session_state.game_state)
        st.rerun()

    # --- SIDEBAR ET HISTORIQUE ---
    with st.sidebar:
        st.subheader("Historique du Dernier Tour")
        for hist_item in p["history"]:
            st.caption(hist_item)
        
        st.subheader("Chat du Jeu")
        chat_msg = st.text_input("Votre message:", key="chat_input")
        if st.button("Envoyer", key="send_chat_msg") and chat_msg:
            update_game_chat(st.session_state.game_id, player_name, chat_msg)
            st.rerun() # Remplacement de st.experimental_rerun()
            
        st.markdown("---")
        st.subheader("Avancement du Tour")
        players_ready = st.session_state.game_state.get('players_ready', {})
        for name, ready in players_ready.items():
            st.caption(f"{name}: {'✅ PRÊT' if ready else '⏳ EN COURS'}")
            
        st.markdown("---")
        if st.session_state.game_state["host_name"] == player_name:
            if st.button("▶️ Forcer l'Avancement du Tour (ADMIN)", type="warning", use_container_width=True):
                handle_turn_advance(st.session_state.game_id, is_forced=True)
                
# ---------------- LOGIQUE DE CONTRÔLE PRINCIPALE ----------------

def handle_turn_advance(game_id, is_forced=False):
    """Vérifie si tous les joueurs sont prêts et simule le tour si oui."""
    
    game_state = st.session_state.game_state
    
    human_players_ready_count = sum(game_state["players_ready"].values())
    total_human_players = len(game_state["players_ready"])
    
    if human_players_ready_count == total_human_players or is_forced:
        
        # --- SIMULATION ---
        st.info("Simulation du tour en cours...")
        
        # S'assurer que tous les humains sont dans le dictionnaire des actions (même s'ils n'ont rien soumis)
        human_players_names = [name for name in game_state["players_ready"].keys()]
        for name in human_players_names:
            if name not in game_state["actions_this_turn"]:
                # Utiliser les actions par défaut (prix inchangés, etc.)
                player_entity = next(p for p in game_state["players"] if p["name"] == name)
                game_state["actions_this_turn"][name] = {"prices": player_entity["prices"]}

        new_game_state = simulate_turn_streamlit(game_state, game_state["actions_this_turn"])
        
        # Mise à jour et sauvegarde de l'état
        st.session_state.game_state = new_game_state
        st.session_state.game_state["actions_this_turn"] = {} # Réinitialiser pour le nouveau tour
        
        save_game_state_to_db(game_id, st.session_state.game_state)
        st.success(f"Tour {new_game_state['turn'] - 1} terminé. Tour {new_game_state['turn']} lancé!")
        
        # Afficher le classement après le tour
        show_leaderboard(st.session_state.game_state)
        
        st.rerun() 
        
    elif game_state["game_status"] == 'in_progress':
        st.sidebar.warning(f"En attente de {total_human_players - human_players_ready_count} joueur(s) humain(s).")


def show_leaderboard(game_state):
    """Affiche le classement actuel des joueurs."""
    st.subheader(f"🏆 Classement - Fin du Tour {game_state['turn'] - 1}")
    
    # Calculer la richesse nette (Money + Assets - Loan)
    player_stats = []
    for p in game_state["players"]:
        net_worth = p["money"] + calculate_asset_value(p["trucks"]) - p["loan"]
        player_stats.append({
            "Entreprise": p["name"],
            "Richesse Nette": net_worth,
            "Trésorerie": p["money"],
            "Dette": p["loan"],
            "Réputation": f"{p['reputation']:.2f}",
            "Actif": "Oui" if p["active"] else "Non"
        })
        
    df = pd.DataFrame(player_stats)
    df = df.sort_values(by="Richesse Nette", ascending=False)
    
    st.dataframe(df.style.format({
        "Richesse Nette": "€ {:,.0f}",
        "Trésorerie": "€ {:,.0f}",
        "Dette": "€ {:,.0f}"
    }).applymap(lambda x: 'background-color: #f7a8a8' if x == 'Non' else '', subset=['Actif']), 
    hide_index=True, use_container_width=True)

    if not any(p['active'] for p in game_state["players"] if p['is_human']):
        st.session_state.game_state['game_status'] = 'finished'
        save_game_state_to_db(st.session_state.game_id, st.session_state.game_state)
        st.balloons()
        st.error("LA PARTIE EST TERMINÉE. Tous les joueurs humains sont en faillite ou liquidés.")
        

# ---------------- VUES DE NAVIGATION ----------------

def afficher_page_accueil():
    st.title("🌐 Jeu de Logistique Multijoueur")
    st.markdown("Choisissez de **Créer** une nouvelle partie ou de **Rejoindre** une partie existante.")

    st.session_state.current_user_name = st.text_input("Votre Nom/Nom d'Entreprise (ex: Ent. A)", key="main_user_name")
    
    if st.session_state.current_user_name:
        col_create, col_join = st.columns(2)
        
        with col_create:
            st.subheader("Créer une Partie")
            st.markdown("Vous deviendrez le **Contrôleur** de la partie.")
            num_ia = st.slider("Nombre d'IA concurrentes initiales", 1, 5, 2, key="num_ia_init")
            host_participates = st.checkbox("Je participe également en tant que joueur", value=True, key="host_participates")
            
            if st.button("🚀 Créer et Lancer Lobby", type="primary", use_container_width=True):
                initialize_game_state(st.session_state.current_user_name, num_ia, host_participates)
                st.session_state.page = "LOBBY"
                st.rerun()

        with col_join:
            st.subheader("Rejoindre une Partie")
            game_id_input = st.text_input("Entrez l'ID de la Partie:", key="join_game_id")
            
            if st.button("🔗 Rejoindre la Partie", type="secondary", use_container_width=True) and game_id_input:
                loaded_state = load_game_state_from_db(game_id_input)
                
                if loaded_state:
                    st.session_state.game_id = game_id_input
                    st.session_state.game_state = loaded_state
                    
                    if loaded_state.get('game_status') == 'in_progress':
                        # Pour les joueurs qui rejoignent après le lancement
                        st.error("Impossible de rejoindre une partie déjà en cours.")
                        
                    elif st.session_state.current_user_name in [p['name'] for p in loaded_state.get('players', [])]:
                        st.session_state.my_name = st.session_state.current_user_name
                        st.session_state.page = "JEU" # Si déjà accepté et en cours
                        st.success("Reconnexion réussie!")
                        st.rerun()
                        
                    elif st.session_state.current_user_name in loaded_state.get('pending_players', []):
                        st.info("Votre demande est en attente d'approbation par le contrôleur.")
                        st.session_state.my_name = st.session_state.current_user_name
                        st.session_state.page = "WAITING"
                        st.rerun()
                        
                    else:
                        # Nouvelle demande d'adhésion
                        st.session_state.game_state['pending_players'].append(st.session_state.current_user_name)
                        st.session_state.my_name = st.session_state.current_user_name
                        save_game_state_to_db(game_id_input, st.session_state.game_state)
                        st.session_state.page = "WAITING"
                        st.success("Demande d'adhésion envoyée. En attente de l'approbation du contrôleur.")
                        st.rerun()
                else:
                    st.error("ID de partie non valide ou non trouvé.")

def show_waiting_room(player_name, game_id):
    st.title("Salle d'Attente")
    st.info(f"ID de la Partie: **{game_id}** | Votre nom: **{player_name}**")
    st.warning("Votre demande d'adhésion est en attente de l'approbation du contrôleur (l'hôte).")
    st.caption("Une fois accepté, cette page sera mise à jour automatiquement.")

    if st.button("Actualiser le Statut", type="secondary"):
        loaded_state = load_game_state_from_db(game_id)
        if loaded_state:
            st.session_state.game_state = loaded_state
            
            if player_name in [p['name'] for p in loaded_state.get('players', [])]:
                st.success("🎉 Vous avez été accepté! Préparez-vous à jouer.")
                st.session_state.page = "JEU"
                st.rerun()
            elif loaded_state.get('game_status') == 'in_progress':
                st.error("La partie a démarré sans vous. Impossible de rejoindre.")
                st.session_state.page = "MENU"
                st.rerun()
            else:
                st.info("Toujours en attente d'approbation...")
        else:
            st.error("Impossible de charger l'état du jeu. L'ID de la partie est peut-être incorrect.")

def main():
    """Contrôleur principal de l'application."""
    
    if 'page' not in st.session_state:
        st.session_state.page = "MENU"
    if 'game_state' not in st.session_state:
        st.session_state.game_state = {}
    
    # Barre latérale de synchronisation pour les joueurs non-host
    if st.session_state.page != "MENU" and st.session_state.get('game_id') and st.session_state.get('my_name') != st.session_state.game_state.get('host_name'):
        with st.sidebar:
            st.header("Synchronisation")
            st.info(f"ID: **{st.session_state.game_id}**")
            if st.button("Actualiser le Statut du Jeu", use_container_width=True):
                sync_game_state(st.session_state.game_id)

    # Logique d'avancement de tour automatique
    if st.session_state.get('game_state', {}).get('game_status') == 'in_progress' and st.session_state.get('my_name') == st.session_state.game_state.get('host_name'):
        handle_turn_advance(st.session_state.game_id)

    # Affichage des pages
    if st.session_state.page == "MENU":
        afficher_page_accueil()
    
    elif st.session_state.page == "LOBBY":
        show_lobby_host(st.session_state.game_id, st.session_state.my_name)

    elif st.session_state.page == "WAITING":
        show_waiting_room(st.session_state.my_name, st.session_state.game_id)
        
    elif st.session_state.page == "JEU":
        if st.session_state.game_state["host_name"] == st.session_state.my_name:
            # L'hôte est aussi le joueur qui déclenche la vue principale de jeu
            show_game_interface_human(st.session_state.my_name)
        else:
            # Pour les joueurs qui rejoignent (non-host)
            show_game_interface_human(st.session_state.my_name)
            
    elif st.session_state.page == "FINISHED":
        st.title("Partie Terminée")
        st.success("Le jeu est arrivé à sa conclusion.")
        show_leaderboard(st.session_state.game_state)
        if st.button("Retour au Menu Principal"):
            st.session_state.page = "MENU"
            st.rerun()

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Jeu de Logistique (Multi)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
