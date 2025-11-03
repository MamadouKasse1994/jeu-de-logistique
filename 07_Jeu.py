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
        # Ceci est nécessaire pour que le code soit runnable dans l'environnement de l'assistant
        # Dans un environnement Streamlit réel, cela devrait être un st.error + st.stop()
        return None # Retourne None si les secrets ne sont pas disponibles

supabase: Client = init_supabase()

# ---------------- FONCTIONS DE SYNCHRONISATION MULTIJOUEUR ----------------

def to_serializable(obj):
    """Gère la conversion d'objets non-JSON-compatibles (uuid, datetime, etc.) en str."""
    if isinstance(obj, uuid.UUID) or isinstance(obj, datetime.datetime):
        return str(obj)
    # Si c'est un dict ou une liste, deepcopy va s'en charger.
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_game_state_to_db(game_id, game_state):
    """Sauvegarde l'état complet du jeu dans Supabase."""
    if not supabase: return # Évite les erreurs si Supabase n'est pas initialisé
    
    game_state_data = deepcopy(game_state)

    # On isole uniquement les clés pertinentes pour le jeu
    keys_to_save = ['game_id', 'turn', 'market_trend', 'backlog_packages', 'event_history',
                    'current_event', 'players', 'num_ia_players', 'host_name',
                    'actions_this_turn', 'players_ready', 'game_ready',
                    'game_status', 'pending_players', 'host_participates']

    state_to_save = {k: game_state_data.get(k) for k in keys_to_save if k in game_state_data}

    # Sérialisation forcée utilisant la fonction utilitaire to_serializable
    data_to_save = {
        "game_id": game_id,
        "state_json": json.dumps(state_to_save, default=to_serializable),
        "turn": game_state_data.get('turn', 1), # Accès direct à game_state_data
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
    if not supabase: return None # Évite les erreurs si Supabase n'est pas initialisé

    try:
        response = supabase.table("games").select("state_json").eq("game_id", game_id).single().execute()
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'état du jeu depuis la base de données: {e}")
        return None

    if response.data and response.data["state_json"]:
        try:
            loaded_state_data = json.loads(response.data["state_json"])
            return loaded_state_data
        except json.JSONDecodeError:
            st.error("Erreur de décodage JSON de l'état du jeu chargé.")
            return None
    else:
        # Game ID not found
        return None

def update_session_from_db(loaded_state_data):
    """Updates st.session_state.game_state from the loaded data."""
    # Preserve user-specific data
    my_name = st.session_state.get('my_name')
    user_name = st.session_state.get('current_user_name') # Keep track of the input field name if needed
    game_id = st.session_state.get('game_id') # Preserve the game ID the user tried to join

    # Update the nested game_state
    st.session_state.game_state = loaded_state_data

    # Restore user-specific data if they were overwritten
    st.session_state.my_name = my_name
    st.session_state.current_user_name = user_name
    st.session_state.game_id = game_id # Ensure game_id at root is the one we tried to load

    st.success("État du jeu synchronisé avec succès!")

def sync_game_state(game_id):
    """Forces loading game state from DB and triggers a rerun."""
    loaded_state = load_game_state_from_db(game_id)
    if loaded_state:
          update_session_from_db(loaded_state)
          # No need to show success message here, it's in update_session_from_db
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
        "timestamp": time.strftime("%H:%M:%S", time.localtime())
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
    # Créer un ensemble des lettres utilisées ou à éviter
    used_initials = {name.split(' ')[1] for name in existing_names if len(name.split(' ')) > 1}
    
    # Itérer sur les lettres disponibles
    ia_letters = itertools.cycle('BCDEFGHIJKLMNOPQRSTUVWXYZ') 
    
    for i in range(num_ia):
        new_name = ""
        while True:
            letter = next(ia_letters)
            # Éviter A si l'hôte est Ent. A (déjà géré dans l'appel initial)
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
        "num_ia_players": num_ia_players, # Initial count
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
        host_entity_name = host_player_name # ex: "Ent. A"
        host_entity = create_player_entity(host_entity_name, True)
        game_state_data["players"].append(host_entity)
        game_state_data["players_ready"][host_player_name] = False
        existing_names.append(host_entity_name)
    else:
        # L'hôte ne joue pas, donc une IA prend sa place dans la boucle de jeu
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

            # Action: Accepter le joueur
            if col_accept.button("✅ Accepter", key=f"accept_{player_name}"):
                new_player_entity = create_player_entity(player_name, True)
                st.session_state.game_state["players"].append(new_player_entity)
                st.session_state.game_state["players_ready"][player_name] = False
                st.session_state.game_state["pending_players"].remove(player_name)

                save_game_state_to_db(game_id, st.session_state.game_state)
                st.success(f"Joueur {player_name} accepté. Vous devez lancer le jeu.")
                st.rerun()

            # Action: Rejeter le joueur
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
    
    # Déterminer si l'hôte est joueur ou IA Host
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
                st.session_state.game_state["num_ia_players"] += 1 # Increment total IA count

            save_game_state_to_db(game_id, st.session_state.game_state)
            st.success(f"{num_new_ia} IA(s) ajoutée(s) avec succès.")
            st.rerun()
        else:
            st.info("Entrez un nombre d'IA supérieur à 0 pour ajouter.")

    st.markdown("---")

    # 4. Lancement de la partie
    st.subheader("Lancer la Partie")
    # Check if there is at least one human player (either the host or an accepted player)
    human_players_exist = any(p['is_human'] for p in st.session_state.game_state.get('players', []))
    disable_start = not human_players_exist # Disable start if no human players

    if st.button("▶️ Lancer la Partie Maintenant", type="primary", disabled=disable_start):
        st.session_state.game_state['game_status'] = 'in_progress'
        # On s'assure que tout le monde est réinitialisé avant le tour 1
        human_players_entities = [p for p in st.session_state.game_state["players"] if p['is_human']]
        st.session_state.game_state["players_ready"] = {p["name"]: False for p in human_players_entities}

        save_game_state_to_db(game_id, st.session_state.game_state)
        st.rerun()

    if not human_players_exist:
        st.warning("Le lancement de la partie est désactivé car il n'y a aucun joueur humain.")
    st.caption("Une fois lancé, de nouveaux joueurs ne pourront plus rejoindre.")


# ---------------- FONCTIONS DE CALCUL (Inchangées) ----------------

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
    # Attractiveness is inversely proportional to the square of the total score (lower score = more attractive)
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
        
        # Effet global sur la réputation/coûts pour les non-couverts sera appliqué plus tard
        
    else:
        game_state["current_event"] = {"name": "Aucun", "text": "Un tour normal.", "type": "None"}
        
# ---------------- LOGIQUE D'IA ----------------

def get_ia_actions(player_data):
    """Détermine les actions d'un joueur IA basé sur une stratégie simple."""
    actions = {}
    
    # Stratégie de prix: baisser légèrement si l'IA est en difficulté ou augmenter si bonne réputation
    new_prices = deepcopy(player_data["prices"])
    
    # 1. Gestion de la faillite (Vente d'urgence)
    if not player_data["active"]:
        actions["sell_trucks"] = {}
        # Vendre le camion le plus ancien et le plus cher
        trucks_sorted = sorted(player_data["trucks"], key=lambda t: t["age"] * t["price"], reverse=True)
        if trucks_sorted:
            model_id = trucks_sorted[0]["id"]
            actions["sell_trucks"][model_id] = 1
        return actions # Fin des actions si en faillite

    # 2. Gestion des prix: Réputation élevée -> Prix élevés, sinon bas pour capter le marché
    price_mod = 0.98 if player_data["reputation"] < 1.0 else 1.02
    for t in COLIS_TYPES:
        # L'IA est plus agressive sur les prix des colis de faible valeur (A, B)
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
    # Sinon, R&D Carburant/Cyber pour se protéger
    elif player_data["money"] > money_threshold * 1.5 and player_data["rd_investment_type"] == "Aucun":
        actions["rd_type"] = random.choice(["Carburant", "CyberSécurité"])
    else:
        actions["rd_type"] = "Aucun"
        
    # Achat de camions si la capacité est faible ou s'il y a de l'argent
    if current_capacity < 1500 and player_data["money"] > 60000:
        model_to_buy = random.choice([TRUCK_MODELS[0], TRUCK_MODELS[1]])
        actions["buy_trucks"] = {model_to_buy["id"]: 1}
    elif current_capacity < 3000 and player_data["money"] > 100000:
        model_to_buy = TRUCK_MODELS[2]
        actions["buy_trucks"] = {model_to_buy["id"]: 1}

    # 4. Gestion des employés: ajouter un employé pour chaque tranche de 1000 de capacité
    target_employees = max(5, int(current_capacity / 1000))
    emp_delta = target_employees - player_data["employees"]
    if abs(emp_delta) > 1: # N'ajuster que par 1
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
    
    # 1. Actions IA (Déjà implémentées par l'utilisateur, vérification du calcul de capacité)
    for i, p in enumerate(game_state["players"]):
        if not p["is_human"]:
            p_cap = calculate_player_capacity(p)
            p["total_capacity"] = p_cap
            ia_action = get_ia_actions(p)
            actions_dict[p["name"]] = ia_action
    
    market_capacity_demand = generate_client_orders(game_state) 
    
    # --- DISTRIBUTION DES COLIS ---
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
                for model_id, qty in action["sell_trucks"].items():
                    if qty > 0:
                        trucks_to_sell = [t for t in p["trucks"] if t.get("id") == model_id][:qty]
                        for truck in trucks_to_sell:
                            if truck in p["trucks"]:
                                p["trucks"].remove(truck)
                                current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10) 
                                resale = int(max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                                p["money"] += resale
                                p["history"].append(f"Vente (Faillite): {truck['id']} (+{resale:,} €)".replace(",", " "))
            
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
                        p["history"].append(f"Achat: {qty}x {model['id']} (-{cost:,} €)".replace(",", " "))
                    else:
                        p["history"].append(f"Achat {qty}x {model['id']} refusé: fonds insuffisants.")
        
        if "sell_trucks" in action and p["active"]:
            for model_id, qty in action["sell_trucks"].items():
                if qty > 0:
                    trucks_to_sell = [t for t in p["trucks"] if t.get("id") == model_id][:qty]
                    for truck in trucks_to_sell:
                        if truck in p["trucks"]:
                            p["trucks"].remove(truck)
                            current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
                            resale = int(max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                            p["money"] += resale
                            p["history"].append(f"Vente: {truck['id']} (+{resale:,} €)".replace(",", " "))
        
        # D. Publicité
        pub_type = action.get("pub_type", "Aucun")
        if pub_type != "Aucun":
            if pub_type == "Locale": cost, rep_inc = (5000, 0.06)
            elif pub_type == "Nationale": cost, rep_inc = (12000, 0.12)
            elif pub_type == "Globale": cost, rep_inc = (25000, 0.25)
            else: cost, rep_inc = (0, 0)
            
            if cost > 0 and p["money"] >= cost:
                p["money"] -= cost
                p["reputation"] = min(5.0, p["reputation"] * (1 + rep_inc))
                p["history"].append(f"Publicité {pub_type}: Réputation +{rep_inc*100:.0f}% (-{cost:,} €)".replace(",", " "))
            elif cost > 0:
                p["history"].append(f"Publicité {pub_type} refusée: fonds insuffisants.")

        # E. Employés (suite de la fonction coupée)
        if "emp_delta" in action and action["emp_delta"] != 0:
            delta = action["emp_delta"]
            if delta > 0:
                # Embauche
                p["employees"] += delta
                p["history"].append(f"Embauche: {delta} employé(s).")
            elif delta < 0 and p["employees"] + delta >= 0:
                # Licenciement
                indemnity = abs(delta) * INDEMNITY_PER_EMP
                if p["money"] >= indemnity:
                    p["money"] -= indemnity
                    p["employees"] += delta
                    p["history"].append(f"Licenciement: {abs(delta)} employé(s) (-{indemnity:,} € d'indemnité).".replace(",", " "))
                else:
                    p["history"].append(f"Licenciement refusé: Fonds insuffisants pour les indemnités.")
            elif p["employees"] + delta < 0:
                p["history"].append("Licenciement refusé: Trop peu d'employés restants.")
        
        # --- PHASE DE RÉSULTATS (REVENUS/DÉPENSES) ---
        
        p["income"] = 0
        p["expenses"] = 0
        
        # 1. Revenus
        total_revenue = 0
        delivered = {t: 0 for t in COLIS_TYPES}
        
        if p["active"]:
            player_capacity_allocation = allocation_capacity.get(p["name"], {})
            
            for t in COLIS_TYPES:
                cap_delivered = player_capacity_allocation.get(t, 0)
                colis_size = CAPACITY_UNITS_PER_COLIS.get(t, 1.0)
                packages_delivered = int(cap_delivered / colis_size)
                
                # Appliquer l'effet d'événement pour Logistique si couvert
                revenue_mod = 1.0
                if current_event["type"] == "Logistique" and p["rd_investment_type"] == "Logistique":
                    revenue_mod = current_event.get("market_bonus_covered", 1.0)
                
                revenue = packages_delivered * p["prices"].get(t, 0) * revenue_mod
                total_revenue += revenue
                delivered[t] = packages_delivered
                p["delivered_packages_total"][t] += packages_delivered
                
            p["income"] = int(total_revenue)
            p["money"] += p["income"]
            p["history"].append(f"Revenus du transport: +{p['income']:,} € ({sum(delivered.values())} colis livrés)".replace(",", " "))
        
        # 2. Dépenses Fixes et Variables
        
        # Appliquer l'effet d'événement de coût si nécessaire
        cost_mod = 1.0
        if current_event["type"] == "Carburant":
            if p["rd_investment_type"] == "Carburant":
                cost_mod = current_event.get("cost_protection_covered", 1.0)
            else:
                cost_mod = current_event.get("cost_increase", 1.0)

        total_expenses = 0
        
        # Salaires
        salaries = p["employees"] * SALARY_PER_EMP
        total_expenses += salaries
        p["history"].append(f"Salaires ({p['employees']} employés): -{salaries:,} €".replace(",", " "))
        
        # Entretien des camions
        maintenance = sum(t["maintenance"] for t in p["trucks"]) * cost_mod
        total_expenses += maintenance
        p["history"].append(f"Maintenance des camions: -{int(maintenance):,} €".replace(",", " "))
        
        # Assurances (varie selon l'âge et le modèle)
        insurance = sum(INSURANCE_PER_TRUCK_BASE * (1 + t["age"] * 0.1) for t in p["trucks"]) * cost_mod
        total_expenses += insurance
        p["history"].append(f"Assurances des camions: -{int(insurance):,} €".replace(",", " "))
        
        # Coûts fixes
        fixed_cost_adjusted = FIXED_COSTS * cost_mod
        total_expenses += fixed_cost_adjusted
        p["history"].append(f"Frais fixes (loyer, IT, etc.): -{int(fixed_cost_adjusted):,} €".replace(",", " "))

        # Impôts
        # Le profit avant impôts est calculé, mais ici on applique une estimation basée sur le revenu
        tax_base = max(0, p["income"] * 0.5) 
        taxes = tax_base * TAX_RATE
        total_expenses += taxes
        p["history"].append(f"Impôts (estimé): -{int(taxes):,} €".replace(",", " "))

        # Intérêts du prêt
        loan_interest = p["loan"] * INTEREST_RATE_PER_TURN
        total_expenses += loan_interest
        if p["loan"] > 0:
            p["history"].append(f"Intérêts de prêt: -{int(loan_interest):,} €".replace(",", " "))

        # Frais imprévus (événement Changement de Réglementation)
        if current_event["type"] == "Reglementation":
            unforeseen_cost = (len(p["trucks"]) * 1000) * current_event.get("unforeseen_cost_mod", 1.0)
            total_expenses += unforeseen_cost
            p["history"].append(f"Frais imprévus (Réglementation): -{int(unforeseen_cost):,} €".replace(",", " "))

        p["expenses"] = int(total_expenses)
        p["money"] -= p["expenses"]

        # 3. Application des pénalités d'événement (Réputation/Argent)
        if current_event["type"] == "Cyber":
            rep_penalty = current_event.get("rep_penalty", 1.0)
            if p["rd_investment_type"] != "CyberSécurité":
                rep_penalty = current_event.get("rep_penalty_uncovered", rep_penalty)
            p["reputation"] = max(0.5, p["reputation"] * rep_penalty)
            if p["rd_investment_type"] != "CyberSécurité":
                 p["money"] -= int(p["money"] * 0.1) # Perte de 10% cash si non couvert
                 p["history"].append("Piratage: Perte de 10% du cash et de réputation si non couvert.")


        # 4. Vieillissement et Fin du Tour
        
        # Vieillissement des camions et dégradation de la réputation
        for truck in p["trucks"]:
            truck["age"] += 1
        
        p["reputation"] = max(1.0, p["reputation"] * 0.98) # Décroissance naturelle de la réputation
        
        # Gestion du prêt et Saisie
        if p["loan"] > 0:
            p["loan_age"] += 1
            # Vérification de saisie
            if p["loan_age"] > MAX_LOAN_AGE_BEFORE_SEIZURE:
                asset_val = calculate_asset_value(p["trucks"])
                if p["loan"] / max(1, asset_val) > FAILLITE_RATIO * 0.8: # Saisie si la dette est élevée après 2 tours sans remboursement
                    trucks_sorted_by_value = sorted(p["trucks"], key=lambda t: calculate_asset_value([t]), reverse=True)
                    if trucks_sorted_by_value:
                        truck_seized = trucks_sorted_by_value[0]
                        seizure_value = calculate_asset_value([truck_seized])
                        p["trucks"].remove(truck_seized)
                        p["loan"] = max(0, p["loan"] - seizure_value)
                        p["loan_age"] = 0 # Reset age
                        p["history"].append(f"🚨 Saisie Bancaire! {truck_seized['id']} saisi pour rembourser la dette. Dette réduite de {seizure_value:,} €.".replace(",", " "))
                    else:
                        p["can_recover"] = False # Plus rien à saisir, faillite permanente

        # 5. Check Faillite
        if p["active"] and p["money"] < 0:
            asset_val = calculate_asset_value(p["trucks"])
            if asset_val == 0 or (p["loan"] + abs(p["money"])) / asset_val > FAILLITE_RATIO:
                p["active"] = False
                p["history"].append("💀 FAILLITE DÉCLENCHÉE! L'entreprise est temporairement inactive (doit vendre des actifs ou rembourser).")
            else:
                p["history"].append(f"Solde négatif: -{abs(p['money']):,} €. Évité la faillite grâce aux actifs.".replace(",", " "))

        # Réinitialiser l'investissement R&D à "Aucun" pour le prochain tour
        p["rd_investment_type"] = "Aucun"
        
        game_state["players"][i] = p
    
    # --- PHASE POST-TOUR ---
    game_state["turn"] += 1
    
    # Vider les actions pour le prochain tour
    game_state["actions_this_turn"] = {}
    
    # Réinitialiser les joueurs prêts
    human_players_entities = [p for p in game_state["players"] if p['is_human']]
    game_state["players_ready"] = {p["name"]: False for p in human_players_entities}
    
    # Vérifier la condition de fin de partie (ex: tous les joueurs humains sont liquidés)
    active_human_players = [p for p in game_state["players"] if p['is_human'] and p["active"]]
    if len(active_human_players) == 0 and any(p['is_human'] for p in game_state["players"]):
        game_state["game_status"] = "finished"
        game_state["history"].append("Fin de partie: Tous les joueurs humains ont été liquidés.")

    # Sauvegarde
    save_game_state_to_db(game_state["game_id"], game_state)
    
    return game_state

# ---------------- LOGIQUE D'AFFICHAGE ET D'INTERFACE (Streamlit) ----------------

# Ces fonctions sont nécessaires pour que le code complet s'exécute, même si l'utilisateur ne les a pas demandées.

def show_game_end_screen(game_state):
    """Affiche l'écran de fin de partie."""
    st.title("FIN DE LA PARTIE 🏁")
    st.info(f"ID de la Partie: {game_state['game_id']}")
    
    # Classement final basé sur la valeur nette (Argent + Actifs - Dette)
    ranking = []
    for p in game_state["players"]:
        net_worth = p["money"] + p["asset_value"] - p["loan"]
        ranking.append({"Nom": p["name"], "Valeur Nette": net_worth, "Type": "Humain" if p["is_human"] else "IA"})
        
    df_ranking = pd.DataFrame(ranking).sort_values(by="Valeur Nette", ascending=False).reset_index(drop=True)
    df_ranking.index = df_ranking.index + 1
    
    st.subheader("Classement Final")
    st.dataframe(df_ranking, use_container_width=True)
    
    st.success("Merci d'avoir joué!")

def show_in_progress_game(game_state):
    """Affiche l'écran principal du jeu en cours."""
    st.title(f"Tour {game_state['turn']} - La Course Logistique")
    
    my_name = st.session_state.my_name
    player_entity = next((p for p in game_state["players"] if p["name"] == my_name and p["is_human"]), None)
    
    if not player_entity:
        st.warning(f"Vous ({my_name}) n'êtes pas un joueur actif dans cette partie. Vous êtes en mode spectateur ou l'hôte IA.")
        # Afficher la vue spectateur/hôte IA
        show_spectator_view(game_state)
        return

    # ------------------ VUE JOUEUR HUMAIN ------------------
    
    is_ready = game_state["players_ready"].get(my_name, False)
    
    st.markdown(f"**Joueur Actuel:** {my_name} | **Statut:** {'✅ Prêt' if is_ready else '⌛ En attente d\'actions'}")

    if not player_entity["active"]:
        st.error(f"🚨 Votre entreprise ({my_name}) est en faillite ou liquidée. Vous ne pouvez faire que des actions de récupération.")
        # UI pour la faillite (vente d'actifs) - Simplifié ici
        st.subheader("Actions de Récupération (Faillite)")
        # ... Logique de faillite (voir plus bas)
        
    st.markdown("---")
    
    col_money, col_rep, col_cap = st.columns(3)
    col_money.metric("💰 Argent Disponible", f"{player_entity['money']:,} €".replace(",", " "))
    col_rep.metric("⭐️ Réputation", f"{player_entity['reputation']:.2f}")
    col_cap.metric("🚛 Capacité Totale", f"{player_entity['total_capacity']:,} u.".replace(",", " "))

    st.markdown("---")

    # Si le joueur est prêt, afficher un message d'attente
    if is_ready:
        st.success("✅ Vos actions sont enregistrées! En attente des autres joueurs.")
        if st.button("Modifier mes Actions (Annuler 'Prêt')"):
            st.session_state.game_state["players_ready"][my_name] = False
            save_game_state_to_db(game_state['game_id'], st.session_state.game_state)
            st.rerun()
        return

    # UI d'Action du Joueur
    with st.expander("📝 Enregistrer les Actions pour le Tour", expanded=True):
        
        # Charger les actions précédentes pour pré-remplir l'UI
        current_actions = game_state["actions_this_turn"].get(my_name, {"prices": player_entity["prices"]})

        # Section 1: Prix de Vente
        st.subheader("1. Fixer les Prix de Vente (€ par colis)")
        new_prices = {}
        cols = st.columns(len(COLIS_TYPES))
        for idx, t in enumerate(COLIS_TYPES):
            new_prices[t] = cols[idx].number_input(f"Colis {t}", value=current_actions.get("prices", player_entity["prices"]).get(t, BASE_PRICES[t]), min_value=1, step=5, key=f"price_{t}_{game_state['turn']}")
            
        # Section 2: Investissements et Personnel
        st.subheader("2. Investissements et Opérations")
        
        col_rd, col_emp = st.columns(2)
        
        # R&D
        rd_options = ["Aucun"] + list(R_D_TYPES.keys())
        rd_type = col_rd.selectbox("Recherche & Développement", options=rd_options, key=f"rd_select_{game_state['turn']}", index=rd_options.index(current_actions.get("rd_type", "Aucun")))
        
        # Personnel
        emp_delta = col_emp.number_input(f"Changement d'Employés (Actuel: {player_entity['employees']})", value=current_actions.get("emp_delta", 0), step=1, key=f"emp_delta_{game_state['turn']}")
        
        # Publicité
        pub_options = ["Aucun", "Locale (5k€)", "Nationale (12k€)", "Globale (25k€)"]
        pub_type = st.selectbox("Campagne de Publicité", options=pub_options, key=f"pub_select_{game_state['turn']}", index=pub_options.index(current_actions.get("pub_type", "Aucun")))
        pub_type_simple = pub_type.split(" ")[0] # Simplification

        # Section 3: Camions
        st.subheader("3. Flotte de Camions")
        
        buy_trucks = {}
        sell_trucks = {}
        
        for model in TRUCK_MODELS:
            col_model, col_buy, col_sell = st.columns(3)
            current_qty = len([t for t in player_entity["trucks"] if t["id"] == model["id"]])
            
            col_model.markdown(f"**{model['id']}** ({model['price']:,} €)".replace(",", " "))
            
            buy_qty = col_buy.number_input(f"Acheter (+{model['id']})", min_value=0, value=current_actions.get("buy_trucks", {}).get(model["id"], 0), step=1, key=f"buy_{model['id']}_{game_state['turn']}", label_visibility="collapsed")
            buy_trucks[model["id"]] = buy_qty
            
            sell_qty = col_sell.number_input(f"Vendre (Actuel: {current_qty})", min_value=0, max_value=current_qty, value=current_actions.get("sell_trucks", {}).get(model["id"], 0), step=1, key=f"sell_{model['id']}_{game_state['turn']}", label_visibility="collapsed")
            sell_trucks[model["id"]] = sell_qty
            
        # Section 4: Banque
        st.subheader("4. Banque et Prêts (Dette actuelle: {p['loan']:,} €)".replace(",", " "))
        col_loan_req, col_loan_pay = st.columns(2)
        loan_request = col_loan_req.number_input("Demander un Prêt", min_value=0, value=current_actions.get("loan_request", 0), step=10000, key=f"loan_req_{game_state['turn']}")
        loan_payment = col_loan_pay.number_input("Rembourser un Prêt", min_value=0, max_value=player_entity['money'], value=current_actions.get("loan_payment", 0), step=10000, key=f"loan_pay_{game_state['turn']}")

        # Bouton Final: Valider les Actions
        final_actions = {
            "prices": new_prices,
            "rd_type": rd_type,
            "emp_delta": emp_delta,
            "pub_type": pub_type_simple,
            "buy_trucks": buy_trucks,
            "sell_trucks": sell_trucks,
            "loan_request": loan_request,
            "loan_payment": loan_payment,
        }

        if st.button("🔒 Valider et Marquer comme PRÊT pour le Tour", type="primary", use_container_width=True):
            st.session_state.game_state["actions_this_turn"][my_name] = final_actions
            st.session_state.game_state["players_ready"][my_name] = True
            save_game_state_to_db(game_state['game_id'], st.session_state.game_state)
            st.rerun()

    # Vue d'ensemble et Historique (après les actions pour ne pas encombrer)
    st.markdown("---")
    st.subheader("📊 Résumé et Historique")
    
    df_trucks = pd.DataFrame(player_entity['trucks'])
    if not df_trucks.empty:
        df_trucks_summary = df_trucks.groupby('id').agg(
            Quantité=('id', 'count'),
            Moyenne_Age=('age', 'mean'),
            Vitesse_Moyenne=('speed', 'mean')
        ).reset_index().rename(columns={'id': 'Modèle'})
        st.write("Détail de votre Flotte:")
        st.dataframe(df_trucks_summary, hide_index=True)
    else:
        st.info("Votre flotte est vide.")
        
    st.write("Dernier Journal de Bord:")
    for entry in player_entity["history"][-5:]:
        st.markdown(f"- {entry}")
        
def show_spectator_view(game_state):
    """Affiche la vue pour l'hôte IA ou le joueur non actif."""
    st.title(f"Tour {game_state['turn']} - Vue Spectateur/Hôte")
    st.info("Vous observez la partie.")
    
    active_human_players = [p for p in game_state["players"] if p['is_human']]
    ready_count = sum(game_state["players_ready"].get(p["name"], False) for p in active_human_players)
    total_human_players = len(active_human_players)
    
    st.subheader("Statut du Tour")
    st.metric("Joueurs Humains Prêts", f"{ready_count} / {total_human_players}")
    st.markdown(f"**Événement du Tour:** {game_state['current_event']['name']} - *{game_state['current_event']['text']}*")
    
    # Bouton de lancement du tour (uniquement si tous sont prêts ou si hôte est IA)
    if ready_count == total_human_players and total_human_players > 0:
        if st.button("▶️ Lancer le Tour Suivant", type="primary", use_container_width=True):
            with st.spinner("Simulation en cours..."):
                simulate_turn_streamlit(game_state, game_state["actions_this_turn"])
            st.rerun()
    elif total_human_players == 0:
         st.warning("Aucun joueur humain actif. La partie ne peut pas progresser.")

    # Tableau de bord général
    st.markdown("---")
    st.subheader("Tableau de Bord des Entreprises")
    
    summary = []
    for p in game_state["players"]:
        net_worth = p["money"] + p["asset_value"] - p["loan"]
        summary.append({
            "Entreprise": p["name"],
            "Argent": f"{p['money']:,} €".replace(",", " "),
            "Actifs": f"{p['asset_value']:,} €".replace(",", " "),
            "Dette": f"{p['loan']:,} €".replace(",", " "),
            "V. Nette": f"{net_worth:,} €".replace(",", " "),
            "Réputation": f"{p['reputation']:.2f}",
            "Capacité": f"{p['total_capacity']:,} u.".replace(",", " "),
            "Statut": "Actif" if p["active"] else "Faillite/Liquid.",
            "Type": "Humain" if p["is_human"] else "IA",
        })
        
    df_summary = pd.DataFrame(summary).sort_values(by="V. Nette", key=lambda x: x.str.replace(' €', '').str.replace(' ', '').astype(int), ascending=False)
    st.dataframe(df_summary, hide_index=True, use_container_width=True)

def show_joining_page(game_id):
    """Page d'attente pour les joueurs qui rejoignent."""
    st.title("Rejoindre une Partie")
    st.info(f"ID de la Partie: **{game_id}**")
    
    my_name = st.session_state.my_name
    game_state = st.session_state.game_state
    
    is_pending = my_name in game_state.get("pending_players", [])
    is_accepted = any(p["name"] == my_name and p["is_human"] for p in game_state["players"])
    
    if is_accepted:
        st.success(f"✅ Vous avez été accepté par l'hôte et faites partie de la partie! En attente du lancement...")
    elif is_pending:
        st.warning("⌛ Votre demande a été envoyée. En attente de l'approbation du contrôleur...")
    else:
        st.error("❌ Erreur de statut: Le jeu n'est pas dans un état attendu pour vous.")
        
    st.markdown("---")
    st.subheader("Discussion de Lobby")
    
    chat_messages = load_game_chat(game_id)
    chat_container = st.container(height=300)
    for msg in reversed(chat_messages):
        chat_container.write(f"**{msg['timestamp']} {msg['sender']}**: {msg['message']}")
        
    chat_input = st.text_input("Votre message", key="chat_input")
    if st.button("Envoyer", use_container_width=False):
        if chat_input:
            update_game_chat(game_id, my_name, chat_input)
            # Rafraîchir pour voir le message
            st.rerun()

def main():
    """Fonction principale de l'application Streamlit."""
    
    if 'game_state' not in st.session_state:
        # Initialisation par défaut si Streamlit est démarré sans état
        st.session_state.game_state = {"game_status": "initial", "game_id": None, "host_name": None}
        st.session_state.my_name = "Ent. A"
        st.session_state.game_id = ""
        st.session_state.current_user_name = "Ent. A"

    game_id = st.session_state.game_id
    game_status = st.session_state.game_state.get("game_status", "initial")
    host_name = st.session_state.game_state.get("host_name")
    my_name = st.session_state.my_name

    st.sidebar.title("Paramètres de la Partie")

    # ------------------- SIDEBAR AUTH/SYNC -------------------

    if game_status in ["initial", "lobby_setup"]:
        st.sidebar.subheader("Créer ou Rejoindre")
        
        # UI pour la création ou la jonction
        host_participates_default = st.sidebar.checkbox("L'hôte participe en tant que joueur", value=True)
        player_name = st.sidebar.text_input("Votre Nom d'Entreprise", value=st.session_state.current_user_name, key="current_user_name_input")
        st.session_state.current_user_name = player_name
        
        # Créer une nouvelle partie
        if st.sidebar.button("➕ Créer une Nouvelle Partie"):
            num_ia = st.sidebar.number_input("Nombre d'IA concurrentes", min_value=1, max_value=5, value=2)
            initialize_game_state(player_name, num_ia, host_participates_default)
            st.rerun()

        # Rejoindre une partie existante
        join_id = st.sidebar.text_input("ID de la Partie à Rejoindre", key="join_game_id")
        if st.sidebar.button("🔗 Rejoindre la Partie"):
            loaded_state = load_game_state_from_db(join_id)
            if loaded_state:
                st.session_state.game_id = join_id
                st.session_state.my_name = player_name
                st.session_state.game_state = loaded_state
                
                # Enregistrer la demande de jonction
                if player_name not in loaded_state["pending_players"] and not any(p["name"] == player_name for p in loaded_state["players"]):
                    st.session_state.game_state["pending_players"].append(player_name)
                    save_game_state_to_db(join_id, st.session_state.game_state)
                    st.session_state.game_state["game_status"] = "joining" # Changement de statut local pour l'UI
                    st.rerun()
                elif any(p["name"] == player_name for p in loaded_state["players"]):
                    st.session_state.game_state["game_status"] = "joining" # déjà accepté
                    st.rerun()
                else:
                    st.error("Vous êtes déjà en attente d'approbation.")
            else:
                st.error("ID de partie introuvable.")

    elif game_status in ["lobby", "joining", "in_progress", "finished"]:
        st.sidebar.subheader("Statut du Jeu")
        st.sidebar.markdown(f"**ID:** `{game_id}`")
        st.sidebar.markdown(f"**Hôte:** `{host_name}`")
        st.sidebar.markdown(f"**Vous:** `{my_name}`")
        st.sidebar.markdown(f"**Tour:** `{st.session_state.game_state.get('turn', 1)}`")

        if st.sidebar.button("Actualiser le Statut du Jeu", type="secondary", use_container_width=True):
            sync_game_state(game_id)
            
        if st.sidebar.button("Quitter la Partie", type="danger", use_container_width=True):
             # Réinitialiser la session pour revenir au menu initial
             for key in list(st.session_state.keys()):
                 del st.session_state[key]
             st.rerun()


    # ------------------- MAIN CONTENT -------------------
    
    is_host = (my_name == host_name)
    
    if game_status == "initial":
        st.title("Bienvenue dans La Course Logistique")
        st.info("Utilisez la barre latérale pour Créer ou Rejoindre une partie.")
        st.caption("Ce jeu simule la gestion d'une entreprise de transport en environnement multijoueur (via Supabase).")
    
    elif game_status == "lobby":
        if is_host:
            show_lobby_host(game_id, host_name)
        else:
            show_joining_page(game_id)
            
    elif game_status == "joining":
        show_joining_page(game_id) # Afficher la page d'attente pour le non-hôte
        
    elif game_status == "in_progress":
        if is_host:
            # L'hôte voit la vue spectateur s'il n'est pas un joueur, sinon sa vue joueur
            host_is_player = st.session_state.game_state.get('host_participates', False)
            if host_is_player:
                show_in_progress_game(st.session_state.game_state)
            else:
                show_spectator_view(st.session_state.game_state)
        else:
            # Les autres joueurs voient leur vue joueur
            show_in_progress_game(st.session_state.game_state)
            
    elif game_status == "finished":
        show_game_end_screen(st.session_state.game_state)

if __name__ == '__main__':
    main()
