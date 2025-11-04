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
from supabase import create_client, Client # Requis pour Supabase

# ----------------------------------------------------------------------
#                         CONSTANTES ET CONFIGURATION
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
#                           CONFIGURATION SUPABASE
# ----------------------------------------------------------------------

@st.cache_resource
def init_supabase():
    """Initialise le client Supabase."""
    try:
        # Utilise st.secrets pour une configuration sécurisée
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        # Gère le cas où l'environnement n'a pas les secrets Supabase (pour test local)
        st.warning("⚠️ Supabase non configuré. Le mode multijoueur est désactivé.")
        return None 

SUPABASE_CLIENT: Client = init_supabase()

# ----------------------------------------------------------------------
#                     FONCTIONS DE SYNCHRONISATION DB
# ----------------------------------------------------------------------

def to_serializable(obj):
    """Gère la conversion d'objets non-JSON-compatibles (uuid, datetime, etc.) en str."""
    if isinstance(obj, uuid.UUID) or isinstance(obj, datetime.datetime):
        return str(obj)
    # Gère le cas où l'objet est déjà sérialisable ou n'a pas besoin de l'être
    return obj 

def save_game_state_to_db(game_id, game_state):
    """Sauvegarde l'état complet du jeu dans Supabase."""
    if not SUPABASE_CLIENT: return 
    
    # Prépare les données à sérialiser
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
        # Utilisez upsert pour insérer ou mettre à jour la ligne avec le même game_id
        SUPABASE_CLIENT.table("games").upsert(data_to_save, on_conflict="game_id").execute()
        # st.toast("Jeu sauvegardé dans la base de données.") # Commenté pour éviter le spam
        return True
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde: {e}")
        return False

def load_game_state_from_db(game_id):
    """Charge l'état complet du jeu depuis Supabase."""
    if not SUPABASE_CLIENT: return None

    try:
        # Charger l'état le plus récent
        response = SUPABASE_CLIENT.table("games").select("state_json").eq("game_id", game_id).limit(1).execute()
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

    if response.data and len(response.data) > 0 and response.data[0].get("state_json"):
        try:
            loaded_state_data = json.loads(response.data[0]["state_json"])
            return loaded_state_data
        except json.JSONDecodeError:
            st.error("Erreur de décodage JSON de l'état du jeu chargé.")
            return None
    else:
        return None

def update_session_from_db(loaded_state_data):
    """Met à jour st.session_state.game_state à partir des données chargées."""
    # Conserver les variables locales critiques de la session
    my_name = st.session_state.get('my_name')
    game_id = st.session_state.get('game_id') 

    # Mettre à jour l'état global
    st.session_state.game_state = loaded_state_data

    # Restaurer les variables locales
    st.session_state.my_name = my_name
    st.session_state.game_id = game_id 

    # Mettre à jour le statut du joueur dans l'état du jeu s'il est humain et actif
    if st.session_state.game_state.get('game_status') == 'in_progress':
        if st.session_state.my_name in st.session_state.game_state.get('players_ready', {}):
             is_ready = st.session_state.game_state['players_ready'].get(st.session_state.my_name, False)
             st.session_state.is_ready = is_ready


def sync_game_state(game_id):
    """Force le chargement de l'état du jeu depuis la DB et déclenche un rerun."""
    loaded_state = load_game_state_from_db(game_id)
    
    if loaded_state:
        is_host = st.session_state.get('my_name') == loaded_state.get('host_name')
        
        # Si c'est l'hôte et qu'on est dans le lobby, traiter les demandes de jointure
        if is_host and loaded_state.get('game_status', 'lobby') == 'lobby':
            loaded_state = load_and_process_join_requests(game_id, loaded_state)
            # Sauvegarder l'état mis à jour par les requêtes de jointure
            save_game_state_to_db(game_id, loaded_state) 
            
        update_session_from_db(loaded_state)
        st.toast("Synchronisation réussie.")
        st.rerun() 
    else:
        st.error("Impossible de se synchroniser. Vérifiez l'ID de la partie.")

# --- NOUVELLES FONCTIONS DE GESTION DES DEMANDES DE JOINTURE ---

def submit_join_request(game_id, player_name):
    """Permet à un joueur de soumettre une demande de jointure."""
    if not SUPABASE_CLIENT: return False
    
    try:
        data = {
            "game_id": game_id,
            "player_name": player_name,
            "status": "pending",
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Vérifier si la requête existe déjà
        check = SUPABASE_CLIENT.table("join_requests").select("id").eq("game_id", game_id).eq("player_name", player_name).eq("status", "pending").limit(1).execute()
        if len(check.data) > 0:
            st.info("Votre demande de jointure est déjà en attente d'approbation.")
            return True 
            
        SUPABASE_CLIENT.table("join_requests").insert(data).execute()
        st.success(f"Demande de jointure envoyée pour la partie {game_id}.")
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de la demande de jointure: {e}")
        return False
        
def load_and_process_join_requests(game_id, game_state):
    """[POUR L'HÔTE] Charge les demandes de jointure et met à jour game_state['pending_players']."""
    if not SUPABASE_CLIENT: return game_state
    
    try:
        response = SUPABASE_CLIENT.table("join_requests").select("id, player_name").eq("game_id", game_id).eq("status", "pending").execute()
        requests = response.data if response.data else []
        
        current_pending = set(game_state.get('pending_players', []))
        new_requests = []
        existing_player_names = {p['name'] for p in game_state.get('players', [])}

        for req in requests:
            player_name = req['player_name']
            if player_name not in existing_player_names and player_name not in current_pending:
                new_requests.append(player_name)
                current_pending.add(player_name)
            
        game_state['pending_players'] = list(current_pending)

        # Supprimer les requêtes traitées (pour éviter de les recharger)
        if requests:
            SUPABASE_CLIENT.table("join_requests").delete().eq("game_id", game_id).eq("status", "pending").execute()

        if new_requests:
            st.toast(f"✅ {len(new_requests)} nouvelle(s) demande(s) de jointure.")
        
        return game_state

    except Exception as e:
        st.error(f"Erreur lors du chargement/traitement des demandes de jointure: {e}")
        return game_state

def update_game_chat(game_id, player_name, message):
    """Ajoute un message au chat de la partie."""
    if not SUPABASE_CLIENT: return
    new_message = {
        "game_id": game_id,
        "sender": player_name,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        SUPABASE_CLIENT.table("chat_messages").insert(new_message).execute()
    except Exception as e:
        st.error(f"Erreur lors de l'envoi du message de chat: {e}")

def load_game_chat(game_id):
    """Charge les 10 derniers messages du chat."""
    if not SUPABASE_CLIENT: return []
    try:
        response = SUPABASE_CLIENT.table("chat_messages").select("sender, message, timestamp").eq("game_id", game_id).order("timestamp", desc=True).limit(10).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Erreur lors du chargement du chat: {e}")
        return []

# ----------------------------------------------------------------------
#                           GESTION DE L'ÉTAT DU JEU
# ----------------------------------------------------------------------

def _new_truck(model):
    """Crée une nouvelle instance de camion avec un UUID."""
    new_truck = deepcopy(model)
    new_truck["uuid"] = uuid.uuid4() 
    return new_truck

def generate_player_names(num_ia, existing_names):
    """Génère de nouveaux noms d'IA uniques."""
    names = []
    used_initials = {name.split(' ')[1] for name in existing_names if len(name.split(' ')) > 1}
    
    ia_letters = itertools.cycle('BCDEFGHIJKLMNOPQRSTUVWXYZ') 
    
    for _ in range(num_ia):
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
    """Crée l'état initial du jeu."""
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

    # 1. Ajout de l'hôte/IA Hôte
    if host_participates:
        host_entity = create_player_entity(host_player_name, True)
        game_state_data["players"].append(host_entity)
        game_state_data["players_ready"][host_player_name] = False
        existing_names.append(host_player_name)
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
    st.session_state.screen = 'lobby'
    st.rerun()

# ----------------------------------------------------------------------
#                           FONCTIONS DE CALCUL (Inchangées)
# ----------------------------------------------------------------------

def calculate_player_capacity(player_data):
    """Calcule la capacité totale de livraison effective d'un joueur."""
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
    """Calcule la valeur de revente estimée des camions (actifs)."""
    total_value = 0
    for truck in player_trucks:
        if not isinstance(truck, dict) or 'id' not in truck: continue
        
        current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
        resale = max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value)
        total_value += resale
    return int(total_value)

def poisson_market(base, trend=1.0):
    """Génère la demande de base du marché selon une distribution de Poisson."""
    return int(np.random.poisson(max(0, base * trend)))

def generate_client_orders(game_state):
    """Génère la demande totale du marché (backlog + nouvelle demande + tendance/événements)."""
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
    """Calcule le score d'attractivité concurrentielle d'un joueur pour un type de colis."""
    player_exec_capacity = p["total_capacity"]
    price_score = p["prices"].get(t, BASE_PRICES.get(t, 500)) * 0.4
    rep_score = 800 / max(1, p["reputation"])
    cap_factor = 1000 / (player_exec_capacity + 1)
    total_score = price_score + rep_score + cap_factor
    attractiveness_weight = 1.0 / (total_score * total_score) 
    return attractiveness_weight, player_exec_capacity

def distribute_clients(market_capacity_demand, players, game_state):
    """Alloue la demande du marché aux joueurs en fonction de leur capacité et de leur score."""
    allocation_capacity = {p["name"]: {t: 0 for t in COLIS_TYPES} for p in players}
    current_package_backlog = {t: 0 for t in COLIS_TYPES}
    active_players = [p for p in players if p["active"]]
    
    player_data = {}
    for p in active_players:
        p_cap = calculate_player_capacity(p)
        p["total_capacity"] = p_cap # Mise à jour de la capacité pour le tour
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
        game_state["backlog_packages"][t] = min(50, current_package_backlog.get(t, 0) + packages_unallocated)
        
    return allocation_capacity

def trigger_random_event(game_state):
    """Déclenche un événement aléatoire avec 40% de chance."""
    if random.random() < 0.4: 
        event = random.choice(EVENT_LIST)
        game_state["current_event"] = event
        game_state["event_history"].append(f"Tour {game_state['turn']}: {event['name']} - {event['text']}")
    else:
        game_state["current_event"] = {"name": "Aucun", "text": "Un tour normal.", "type": "None"}
        
# ----------------------------------------------------------------------
#                               LOGIQUE D'IA
# ----------------------------------------------------------------------

def get_ia_actions(player_data):
    """Détermine les actions d'un joueur IA basé sur une stratégie simple."""
    actions = {}
    
    new_prices = deepcopy(player_data["prices"])
    
    if not player_data["active"]:
        actions["sell_trucks"] = {}
        trucks_sorted = sorted(player_data["trucks"], key=lambda t: t["age"] * t["price"], reverse=True)
        if trucks_sorted:
            model_id = trucks_sorted[0]["id"]
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
    
    if player_data["money"] > money_threshold and player_data["rd_boost_log"] < 0.2:
        actions["rd_type"] = "Logistique"
    elif player_data["money"] > money_threshold * 1.5 and player_data["rd_investment_type"] == "Aucun":
        actions["rd_type"] = random.choice(["Carburant", "CyberSécurité"])
    else:
        actions["rd_type"] = "Aucun"
        
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

# ----------------------------------------------------------------------
#                          LOGIQUE DU TOUR DE JEU
# ----------------------------------------------------------------------

def simulate_turn_streamlit(game_state, actions_dict):
    """Exécute un tour de simulation."""
    
    trigger_random_event(game_state)
    current_event = game_state["current_event"]
    event_info = f"🌪️ Événement du Tour: **{current_event['name']}** - {current_event['text']}"
    
    # 1. Actions IA (Capacité pour l'IA et actions IA)
    for p in game_state["players"]:
        if not p["is_human"]:
            p_cap = calculate_player_capacity(p)
            p["total_capacity"] = p_cap
            ia_action = get_ia_actions(p)
            actions_dict[p["name"]] = ia_action
    
    market_capacity_demand = generate_client_orders(game_state) 
    allocation_capacity = distribute_clients(market_capacity_demand, game_state["players"], game_state)

    # --- PHASE D'APPLICATION DES ACTIONS ET CALCUL DES FINANCES ---
    for i, p in enumerate(game_state["players"]):
        
        p["history"] = [event_info]
        actions = actions_dict.get(p["name"], {})
        
        # --- A. Revenues ---
        p["income"] = 0
        p["delivered_this_turn"] = {t: 0 for t in COLIS_TYPES}
        
        for t in COLIS_TYPES:
            capacity_allocated = allocation_capacity.get(p["name"], {}).get(t, 0)
            colis_size = CAPACITY_UNITS_PER_COLIS.get(t, 1.0)
            packages_delivered = int(capacity_allocated / colis_size)
            
            revenue = packages_delivered * p["prices"].get(t, 0)
            p["income"] += revenue
            p["delivered_packages_total"][t] += packages_delivered
            p["delivered_this_turn"][t] = packages_delivered
            
        # --- B. Dépenses et Coûts Fixes ---
        p["expenses"] = 0
        
        # Coûts fixes et salaires
        event_cost_mod = current_event.get("cost_increase", 1.0)
        
        # Protection Carburant (R&D)
        if current_event["type"] == "Carburant" and p["rd_investment_type"] == "Carburant":
            event_cost_mod = current_event.get("cost_protection_covered", 1.0)

        salaries = p["employees"] * SALARY_PER_EMP
        fixed_costs = FIXED_COSTS * event_cost_mod
        p["expenses"] += salaries + fixed_costs
        p["history"].append(f"Frais fixes et Salaires: -{fixed_costs + salaries:,.0f}€ (Facteur coût: {event_cost_mod:.2f})")

        # Entretien des camions
        maintenance_cost = sum(t.get("maintenance", 0) for t in p["trucks"])
        insurance_cost = len(p["trucks"]) * INSURANCE_PER_TRUCK_BASE
        p["expenses"] += maintenance_cost + insurance_cost
        p["history"].append(f"Entretien et Assurances: -{maintenance_cost + insurance_cost:,.0f}€")

        # Remboursement de prêt et intérêts
        if p["loan"] > 0:
            interest = p["loan"] * INTEREST_RATE_PER_TURN
            min_payment = p["loan"] * MIN_LOAN_PAYMENT_RATIO
            payment = min_payment + interest
            p["loan"] -= min_payment
            p["expenses"] += payment
            p["loan_age"] += 1
            p["history"].append(f"Prêt (Intérêts/Principal): -{payment:,.0f}€ (Prêt restant: {p['loan']:,.0f}€)")
        else:
            p["loan_age"] = 0
        
        # --- C. Applications des Actions Spécifiques ---
        
        # Achat/Vente de camions
        # (Logique de transaction et mise à jour de p["trucks"] ici - OMISE POUR LA BREVETÉ MAIS PRÉSENTE DANS LE CODE ORIGINAL)
        
        # R&D (seulement le coût ici)
        rd_type = actions.get("rd_type")
        if rd_type != "Aucun" and rd_type in R_D_TYPES:
            rd_cost = R_D_TYPES[rd_type]["cost"]
            if p["money"] >= rd_cost:
                p["expenses"] += rd_cost
                p["rd_investment_type"] = rd_type
                if rd_type == "Logistique":
                    p["rd_boost_log"] += R_D_TYPES[rd_type]["boost_value"]
                p["history"].append(f"Investissement R&D ({rd_type}): -{rd_cost:,.0f}€")
            else:
                p["history"].append("Échec R&D: Fonds insuffisants.")
                
        # --- D. Application des Modifications de Prix (Mise à jour immédiate pour l'IA) ---
        new_prices_action = actions.get("prices")
        if new_prices_action:
             p["prices"] = new_prices_action

        # --- E. Application des Employés
        emp_delta = actions.get("emp_delta", 0)
        if emp_delta != 0:
            if emp_delta > 0:
                p["employees"] += emp_delta
                p["history"].append(f"Embauche de {emp_delta} employé(s).")
            elif emp_delta < 0:
                release_cost = abs(emp_delta) * INDEMNITY_PER_EMP
                p["employees"] += emp_delta
                p["expenses"] += release_cost
                p["history"].append(f"Licenciement de {-emp_delta} employé(s). Coût: -{release_cost:,.0f}€")


        # --- F. Réputation et Événements ---
        
        # Pénalité de réputation (si événement non couvert)
        if current_event.get("rep_penalty_uncovered") and p["rd_investment_type"] != current_event["type"]:
            p["reputation"] *= current_event["rep_penalty_uncovered"]
            p["history"].append(f"Impact négatif de l'événement {current_event['name']} (non couvert) sur la réputation.")
        elif current_event.get("rep_penalty"): # Impact général
             p["reputation"] *= current_event["rep_penalty"]

        # La réputation se rapproche lentement de 1.0 si > 1.0 (ou s'éloigne si < 1.0)
        p["reputation"] = p["reputation"] * 0.95 + 0.05 * 1.0
        p["reputation"] = max(0.5, min(2.0, p["reputation"]))


        # --- G. Mouvement de Trésorerie ---
        p["money"] += p["income"] - p["expenses"]
        
        # --- H. Fin de Tour et Faillite ---
        p["asset_value"] = calculate_asset_value(p["trucks"])
        
        if p["money"] < 0:
            p["can_recover"] = True # Tentative de recouvrement
            if p["loan"] > 0 and p["loan_age"] >= MAX_LOAN_AGE_BEFORE_SEIZURE:
                p["active"] = False
                p["history"].append("FAILLITE (Prêt non remboursé et au-delà de l'âge limite)!")
            elif abs(p["money"]) > 1.5 * p["asset_value"]: # Dette > 150% des actifs
                p["active"] = False
                p["history"].append("FAILLITE (Dette trop importante par rapport aux actifs)!")
            elif abs(p["money"]) > p["asset_value"] * 0.8:
                p["history"].append("ATTENTION: Votre entreprise est en **danger de faillite** (dette > 80% des actifs). Vendez des actifs ou obtenez un prêt.")
            else:
                 p["history"].append(f"Trésorerie Négative: -{abs(p['money']):,.0f}€. Action de survie requise.")
        else:
            p["can_recover"] = False
            
        game_state["players"][i] = p # Mettre à jour l'état du joueur
        
    game_state["turn"] += 1
    game_state["players_ready"] = {p["name"]: False for p in game_state["players"] if p["is_human"]}
    game_state["actions_this_turn"] = {}
    return game_state

# ----------------------------------------------------------------------
#                             FONCTIONS D'INTERFACE
# ----------------------------------------------------------------------

def render_chat(game_id):
    """Affiche la zone de chat."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat de la Partie")
    
    chat_messages = load_game_chat(game_id)
    chat_container = st.sidebar.container(height=300)
    
    # Afficher les messages dans l'ordre chronologique
    for msg in reversed(chat_messages):
        # Formatage simple du temps (par exemple HH:MM)
        try:
            ts = datetime.datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')).strftime("%H:%M")
        except:
            ts = ""
        
        chat_container.caption(f"**{msg['sender']}** ({ts}):")
        chat_container.write(msg['message'])
        
    # Champ d'entrée pour un nouveau message
    user_message = st.sidebar.text_input("Votre message", key="chat_input")
    if st.sidebar.button("Envoyer", use_container_width=True):
        if user_message:
            update_game_chat(game_id, st.session_state.my_name, user_message)
            # Effacer le champ d'entrée après l'envoi
            st.session_state.chat_input = "" 
            st.rerun()

def show_sidebar(game_id, host_name):
    """Affiche la barre latérale pour tous les joueurs."""
    
    # 1. Infos de base
    st.sidebar.title("LogiSim - Multijoueur")
    st.sidebar.markdown(f"**ID Partie**: `{game_id}`")
    st.sidebar.markdown(f"**Contrôleur**: `{host_name}`")
    st.sidebar.markdown(f"**Votre Nom**: `{st.session_state.my_name}`")
    st.sidebar.markdown(f"**Tour Actuel**: **{st.session_state.game_state.get('turn', 0)}**")
    
    # 2. Bouton de Synchro/Actualisation
    if st.sidebar.button("Actualiser le Statut du Jeu", type="secondary", use_container_width=True):
        sync_game_state(game_id)
        
    # 3. Afficher le chat
    render_chat(game_id)


def show_setup_screen():
    """Affiche l'écran initial pour créer ou rejoindre une partie."""
    st.title("🚛 LogiSim - Simulation de Logistique Multijoueur")
    st.subheader("Bienvenue dans l'interface de connexion.")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Créer une Partie (Hôte)")
        host_name = st.text_input("Votre nom d'utilisateur (Hôte)", key="host_name_input")
        num_ia = st.number_input("Nombre d'IA concurrentes", min_value=1, max_value=10, value=3, key="num_ia_input")
        host_participates = st.checkbox("Je participe en tant qu'entreprise 'Ent. A'", value=True, key="host_participates_check")
        
        if st.button("🚀 Créer la Partie", type="primary", use_container_width=True, disabled=not host_name):
            initialize_game_state(host_name, num_ia, host_participates)

    with col2:
        st.header("Rejoindre une Partie (Joueur)")
        join_game_id = st.text_input("ID de la Partie à rejoindre (ex: GAME-1A2B3C)", key="join_id_input").upper()
        player_name = st.text_input("Votre nom d'utilisateur (Joueur)", key="player_name_input")

        if st.button("🤝 Envoyer Demande de Jointure", type="secondary", use_container_width=True, disabled=not (join_game_id and player_name)):
            if submit_join_request(join_game_id, player_name):
                # Mise à jour de la session pour l'écran d'attente
                st.session_state.my_name = player_name
                st.session_state.game_id = join_game_id
                st.session_state.screen = 'pending'
                st.rerun()

def show_pending_screen(game_id):
    """Écran pour les joueurs qui attendent l'approbation de l'hôte."""
    st.title("⏳ En attente de l'approbation du Contrôleur")
    st.markdown(f"Votre demande de jointure pour la partie **{game_id}** a été envoyée.")
    st.info("Veuillez attendre que le contrôleur vous accepte dans le lobby.")
    st.warning("Cliquez sur **'Actualiser le Statut du Jeu'** dans la barre latérale pour vérifier si vous avez été accepté ou si le jeu a commencé.")

def show_lobby_host():
    """Interface du lobby pour le contrôleur (Host)."""
    
    game_state = st.session_state.game_state
    game_id = game_state['game_id']
    host_name = game_state['host_name']

    st.title("Admin de la Partie et Lobby")
    st.info(f"ID de la Partie: **{game_id}** | Contrôleur: **{host_name}**")

    # 1. Gestion des joueurs en attente
    st.subheader("🚪 Joueurs en Attente d'Approbation")
    pending_players = game_state.get('pending_players', [])

    if pending_players:
        for player_name in pending_players:
            col_name, col_accept, col_reject = st.columns([3, 1, 1])
            col_name.write(f"**{player_name}**")

            if col_accept.button("✅ Accepter", key=f"accept_{player_name}"):
                # Ajout du joueur, retrait des pending
                new_player_entity = create_player_entity(player_name, True)
                game_state["players"].append(new_player_entity)
                game_state["players_ready"][player_name] = False
                game_state["pending_players"].remove(player_name)
                update_game_chat(game_id, "System", f"Le joueur {player_name} a été accepté dans la partie.")
                save_game_state_to_db(game_id, game_state)
                st.success(f"Joueur {player_name} accepté.")
                st.rerun()

            if col_reject.button("❌ Rejeter", key=f"reject_{player_name}"):
                game_state["pending_players"].remove(player_name)
                save_game_state_to_db(game_id, game_state)
                st.warning(f"Joueur {player_name} rejeté.")
                st.rerun()
    else:
        st.info("Aucun joueur en attente pour l'instant. Utilisez 'Actualiser' si vous attendez des joueurs.")

    st.markdown("---")

    # 2. Liste des joueurs acceptés
    st.subheader("👥 Joueurs Participants")
    all_players = game_state.get('players', [])
    
    if all_players:
        df_players = pd.DataFrame([
            {"Nom": p['name'], "Type": "Humain" if p['is_human'] else "IA", "Rôle": "Contrôleur/Joueur" if p["name"] == host_name and p["is_human"] else ("Joueur" if p["is_human"] else "IA")}
            for p in all_players
        ])
        st.dataframe(df_players, hide_index=True, use_container_width=True)

    st.markdown("---")

    # 3. Lancement de la partie
    st.subheader("Lancer la Partie")
    human_players_exist = any(p['is_human'] for p in all_players)
    disable_start = not human_players_exist 

    if st.button("▶️ Lancer la Partie Maintenant", type="primary", disabled=disable_start):
        game_state['game_status'] = 'in_progress'
        human_players_entities = [p for p in game_state["players"] if p['is_human']]
        game_state["players_ready"] = {p["name"]: False for p in human_players_entities}
        update_game_chat(game_id, "System", f"La partie a été lancée par le contrôleur {host_name}.")
        save_game_state_to_db(game_id, game_state)
        st.rerun()

    if not human_players_exist:
        st.warning("Le lancement est désactivé car il n'y a aucun joueur humain (l'hôte doit participer ou des joueurs doivent être ajoutés).")


def show_in_progress_game():
    """Affiche l'écran de jeu actif."""
    
    game_state = st.session_state.game_state
    my_name = st.session_state.my_name
    game_id = game_state['game_id']
    
    st.title(f"LogiSim - Tour {game_state['turn']}")
    
    # Trouver le joueur actif
    my_player_entity = next((p for p in game_state["players"] if p["name"] == my_name), None)

    if not my_player_entity:
        st.error("Votre entreprise n'existe plus dans cette partie. Vous êtes un observateur.")
        return

    # AFFICHER L'ÉTAT ACTUEL DE LA PARTIE
    col_money, col_rep, col_loan, col_cap = st.columns(4)
    col_money.metric("Trésorerie", f"{my_player_entity['money']:,.0f} €", delta=my_player_entity['income'] - my_player_entity['expenses'])
    col_rep.metric("Réputation", f"{my_player_entity['reputation']:.2f}", delta=0)
    col_loan.metric("Prêt Restant", f"{my_player_entity['loan']:,.0f} €")
    col_cap.metric("Capacité Effective", f"{my_player_entity['total_capacity']:,.0f} u")

    st.markdown("---")
    
    st.subheader(f"Actions de l'Entreprise : {my_name}")
    
    # Vérification si les actions ont déjà été soumises
    is_ready = st.session_state.is_ready if 'is_ready' in st.session_state else game_state['players_ready'].get(my_name, False)
    
    if is_ready:
        st.success("✅ Vos actions pour ce tour ont été soumises. En attente des autres joueurs...")
        st.info(f"Joueurs non prêts : {len([name for name, ready in game_state['players_ready'].items() if not ready and name != my_name])}")
        
        # LOGIQUE HÔTE : Passage au tour suivant
        if my_name == game_state['host_name'] and all(game_state['players_ready'].values()):
            if st.button("▶️ Lancer le Tour Suivant (Host)"):
                # Simuler le tour et mettre à jour l'état
                new_game_state = simulate_turn_streamlit(game_state, game_state['actions_this_turn'])
                save_game_state_to_db(game_state['game_id'], new_game_state)
                st.toast("Nouveau tour lancé!")
                st.session_state.is_ready = False
                st.rerun()
        return

    # --- Écran de Saisie des Actions ---
    
    # 1. Définition des prix
    st.markdown("#### 1. Définir les Prix de Livraison")
    new_prices = deepcopy(my_player_entity['prices'])
    cols = st.columns(len(COLIS_TYPES))
    for i, t in enumerate(COLIS_TYPES):
        new_prices[t] = cols[i].number_input(f"Prix Colis {t} (€)", value=new_prices[t], min_value=10, key=f"price_{t}")

    # 2. Achat/Vente de camions
    st.markdown("#### 2. Gestion de la Flotte")
    # ... UI pour l'achat de camions
    
    # 3. R&D
    st.markdown("#### 3. Investissement R&D")
    rd_choice = st.selectbox("Choisir l'investissement R&D (coût ~20k€)", ["Aucun"] + list(R_D_TYPES.keys()), key="rd_choice")

    # 4. Finalisation et Soumission
    st.markdown("---")
    
    if st.button("Soumettre les Actions du Tour", type="primary", use_container_width=True):
        
        # Construire l'objet d'actions
        current_actions = {
            "prices": new_prices,
            "rd_type": rd_choice,
            # Ajouter les actions de camions, employés, etc. ici
        }
        
        # Mettre à jour l'état dans la session
        player_index = next((i for i, p in enumerate(game_state["players"]) if p["name"] == my_name), -1)
        if player_index != -1:
            # L'entité joueur est mise à jour localement avec les prix décidés pour ce tour
            game_state["players"][player_index]["prices"] = new_prices 
        
        game_state['actions_this_turn'][my_name] = current_actions
        game_state['players_ready'][my_name] = True
        st.session_state.is_ready = True # Mise à jour locale pour le rafraîchissement
        
        # Sauvegarder l'état complet dans la DB
        save_game_state_to_db(game_id, game_state)
        st.toast("Actions soumises! En attente des autres joueurs.")
        st.rerun()


# ----------------------------------------------------------------------
#                             FONCTION PRINCIPALE
# ----------------------------------------------------------------------

def main():
    """Fonction principale de l'application Streamlit."""
    
    # Initialisation de l'état de session si c'est le premier lancement
    if 'screen' not in st.session_state:
        st.session_state.screen = 'setup'
    if 'game_state' not in st.session_state:
        st.session_state.game_state = {}
    if 'my_name' not in st.session_state:
        st.session_state.my_name = None
    if 'game_id' not in st.session_state:
        st.session_state.game_id = None
        
    game_id = st.session_state.game_id
    host_name = st.session_state.game_state.get('host_name')

    # --- Affichage de la Barre Latérale (dès que le jeu est initié/rejoint) ---
    if game_id:
        show_sidebar(game_id, host_name)
        
    # --- LOGIQUE D'AFFICHAGE SELON L'ÉTAT ---
    
    current_status = st.session_state.game_state.get('game_status', 'lobby')
    
    if st.session_state.screen == 'setup':
        show_setup_screen()
        
    elif st.session_state.screen == 'pending':
        # Vient de soumettre une demande, doit se synchroniser pour voir s'il est accepté
        if current_status == 'lobby':
            show_pending_screen(game_id)
        elif current_status == 'in_progress':
            # Le joueur a été accepté ou le jeu a été lancé
            st.session_state.screen = 'game'
            st.rerun()
            
    elif st.session_state.screen == 'lobby':
        is_host = st.session_state.my_name == host_name
        
        if is_host and current_status == 'lobby':
            show_lobby_host()
        elif current_status == 'in_progress':
            st.session_state.screen = 'game'
            st.rerun()
        else:
            # Joueur dans le lobby, non hôte
            st.title(f"Lobby - Participer à {game_id}")
            st.info(f"Vous avez rejoint la partie en tant que **{st.session_state.my_name}**. En attente du lancement par l'hôte.")
            st.warning("Utilisez le bouton 'Actualiser le Statut du Jeu' dans la barre latérale.")
            
    elif st.session_state.screen == 'game' and current_status == 'in_progress':
        show_in_progress_game()


if __name__ == '__main__':
    main()
