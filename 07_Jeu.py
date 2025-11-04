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
        # Ceci nécessite que `st.secrets` soit configuré correctement
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        # Placeholder pour l'environnement de l'assistant/test local sans secrets
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
        # CORRECTION du bug PGRST116 (doublons)
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

# --- NOUVELLES FONCTIONS DE GESTION DES DEMANDES DE JOINTURE (HOST/PLAYER) ---

def submit_join_request(game_id, player_name):
    """Permet à un joueur de soumettre une demande de jointure à la collection join_requests."""
    if not supabase: 
        st.error("Le service de base de données est indisponible.")
        return False
    if not game_id or not player_name:
        st.warning("L'ID de partie et le nom du joueur sont requis.")
        return False
        
    try:
        data = {
            "game_id": game_id,
            "player_name": player_name,
            "status": "pending",
            "created_at": datetime.datetime.now().isoformat()
        }
        # Vérifier si la requête existe déjà
        check = supabase.table("join_requests").select("id").eq("game_id", game_id).eq("player_name", player_name).eq("status", "pending").limit(1).execute()
        if len(check.data) > 0:
            st.info("Votre demande de jointure est déjà en attente d'approbation.")
            return True # Déjà en attente
            
        supabase.table("join_requests").insert(data).execute()
        st.success(f"Demande de jointure envoyée pour la partie {game_id}. En attente de l'approbation du contrôleur.")
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de la demande de jointure: {e}")
        return False
        
def load_and_process_join_requests(game_id, game_state):
    """
    [POUR L'HÔTE] Charge les demandes de jointure de la collection 'join_requests', 
    met à jour game_state['pending_players'] et supprime les requêtes de la base.
    """
    if not supabase: return game_state
    
    try:
        # Charger toutes les requêtes en attente pour cette partie
        response = supabase.table("join_requests").select("id, player_name").eq("game_id", game_id).eq("status", "pending").execute()
        requests = response.data if response.data else []
        
        # 1. Mettre à jour la liste pending_players dans game_state
        current_pending = set(game_state.get('pending_players', []))
        new_requests = []
        
        existing_player_names = {p['name'] for p in game_state.get('players', [])}

        for req in requests:
            player_name = req['player_name']
            # N'ajouter que si le joueur n'est pas déjà dans la partie et n'est pas déjà en attente
            if player_name not in existing_player_names and player_name not in current_pending:
                new_requests.append(player_name)
                current_pending.add(player_name)
            
        game_state['pending_players'] = list(current_pending)

        # 2. Supprimer les requêtes traitées de la base (pour éviter de les recharger au prochain sync)
        if requests:
            supabase.table("join_requests").delete().eq("game_id", game_id).eq("status", "pending").execute()

        if new_requests:
            st.toast(f"✅ {len(new_requests)} nouvelle(s) demande(s) de jointure.")
        
        return game_state

    except Exception as e:
        st.error(f"Erreur lors du chargement/traitement des demandes de jointure: {e}")
        return game_state


def sync_game_state(game_id):
    """Forces loading game state from DB and triggers a rerun."""
    loaded_state = load_game_state_from_db(game_id)
    if loaded_state:
        # NOUVEAU: Si nous sommes l'hôte et en phase de lobby, charger les requêtes de jointure
        is_host = st.session_state.get('my_name') == loaded_state.get('host_name')
        if is_host and loaded_state.get('game_status', 'lobby') == 'lobby':
            loaded_state = load_and_process_join_requests(game_id, loaded_state)
            # Sauvegarder immédiatement l'état mis à jour par les requêtes de jointure dans le DB
            save_game_state_to_db(game_id, loaded_state) 
            
        update_session_from_db(loaded_state)
        st.rerun() # Déclencher le rafraîchissement de l'interface
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
    """Crée une nouvelle instance de camion avec un UUID."""
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
        # L'appel à sync_game_state recharge l'état complet du jeu, incluant les requêtes de jointure traitées
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

                # Annonce dans le chat
                update_game_chat(game_id, "System", f"Le joueur {player_name} a été accepté dans la partie.")
                
                save_game_state_to_db(game_id, st.session_state.game_state)
                st.success(f"Joueur {player_name} accepté. Vous devez lancer le jeu.")
                st.rerun()

            if col_reject.button("❌ Rejeter", key=f"reject_{player_name}"):
                st.session_state.game_state["pending_players"].remove(player_name)
                # Note: Le rejet n'est pas sauvegardé dans une base séparée pour le joueur.
                # Le joueur devra resoumettre ou l'hôte devra le communiquer.
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

        update_game_chat(game_id, "System", f"La partie a été lancée par le contrôleur {host_name}.")

        save_game_state_to_db(game_id, st.session_state.game_state)
        st.rerun()

    if not human_players_exist:
        st.warning("Le lancement de la partie est désactivé car il n'y a aucun joueur humain.")
    st.caption("Une fois lancé, de nouveaux joueurs ne pourront plus rejoindre.")


# ---------------- FONCTIONS DE CALCUL ----------------

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
        
        # Décote annuelle de 10%
        current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
        # Valeur minimum de revente
        resale = max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value)
        total_value += resale
    return int(total_value)

def poisson_market(base, trend=1.0):
    """Génère la demande de base du marché selon une distribution de Poisson."""
    return int(np.random.poisson(max(0, base * trend)))

def generate_client_orders(game_state):
    """Génère la demande totale du marché (backlog + nouvelle demande + tendance/événements)."""
    package_orders = {t: 0 for t in COLIS_TYPES}
    
    # 1. Ajout du backlog
    for t in COLIS_TYPES:
        package_orders[t] += game_state["backlog_packages"].get(t, 0)
    
    # 2. Ajout des commandes des clients individuels
    for _ in range(TOTAL_CLIENTS * 2): 
        types_chosen = random.choices(COLIS_TYPES, k=random.randint(1, 3))
        for t in types_chosen:
            package_orders[t] += random.randint(1, 5)

    # 3. Ajout de la demande de masse (Poisson + Tendance)
    for t in COLIS_TYPES:
        package_orders[t] += poisson_market(BASE_DEMANDS.get(t, 0), game_state["market_trend"]) 
        
    # 4. Impact de l'événement (modifie la demande globale)
    if "market_effect" in game_state["current_event"]:
        for t in package_orders:
            package_orders[t] = int(package_orders[t] * game_state["current_event"]["market_effect"])

    # Conversion des colis en unités de capacité demandée
    capacity_required = {
        t: int(package_orders[t] * CAPACITY_UNITS_PER_COLIS.get(t, 1.0)) 
        for t in COLIS_TYPES
    }
    
    return capacity_required

def calculate_competition_score(p, t):
    """Calcule le score d'attractivité concurrentielle d'un joueur pour un type de colis."""
    player_exec_capacity = p["total_capacity"]
    
    # Facteur Prix : Plus le prix est bas, plus le score est bas (bon pour le client)
    price_score = p["prices"].get(t, BASE_PRICES.get(t, 500)) * 0.4
    
    # Facteur Réputation : Plus la réputation est haute, plus le score est bas (bon pour le client)
    rep_score = 800 / max(1, p["reputation"])
    
    # Facteur Capacité : Les clients préfèrent les entreprises avec une grande capacité
    cap_factor = 1000 / (player_exec_capacity + 1)
    
    total_score = price_score + rep_score + cap_factor
    
    # L'attractivité (poids) est inversement proportionnelle au carré du score (favorise les extrêmes)
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
        unit_size = max(1, qty_capacity_remaining // 4) # Tenter de distribuer par blocs
        
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
            
            # Choisir un joueur pondéré par son score
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
                    # Joueur est à pleine capacité pour ce tour, ne peut plus prendre de colis
                    pass
            else:
                # Plus aucun joueur ne peut prendre de capacité restante
                break
                
        # Calcul du backlog
        capacity_unallocated = max(0, qty_capacity_remaining)
        packages_unallocated = int(capacity_unallocated / colis_size)
        game_state["backlog_packages"][t] = min(50, current_package_backlog[t] + packages_unallocated)
        
    return allocation_capacity

def trigger_random_event(game_state):
    """Déclenche un événement aléatoire avec 40% de chance."""
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
        # Vendre le camion le plus âgé/moins performant (approche simplifiée)
        trucks_sorted = sorted(player_data["trucks"], key=lambda t: t["age"] * t["price"], reverse=True)
        if trucks_sorted:
            model_id = trucks_sorted[0]["id"]
            # En IA, on vend juste le premier camion trouvé de ce modèle
            truck_uuid = str(trucks_sorted[0]["uuid"])
            actions["sell_trucks"][model_id] = [truck_uuid]
        return actions 

    # 2. Gestion des prix
    # L'IA baisse les prix si la réputation est faible, les augmente si elle est haute
    price_mod = 0.98 if player_data["reputation"] < 1.0 else 1.02
    for t in COLIS_TYPES:
        if t in ["A", "B"]:
            # Les IA avec R&D logistique peuvent se permettre des prix plus bas sur les petits colis
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
    # La capacité est recalculée pour tous les joueurs dans distribute_clients (si actif)
    allocation_capacity = distribute_clients(market_capacity_demand, game_state["players"], game_state)

    # --- PHASE D'APPLICATION DES ACTIONS DÉCIDÉES PAR LES JOUEURS ---
    for i, p in enumerate(game_state["players"]):
        
        p["history"] = [event_info]
        action = actions_dict.get(p["name"], {"prices": p["prices"]}).copy()
        
        p["prices"] = action.get("prices", p["prices"])
        p["rd_boost_log"] = p.get("rd_boost_log", 0) 
        p["rd_investment_type"] = action.get("rd_type", p.get("rd_investment_type", "Aucun")) # Garde l'investissement du tour précédent si non renouvelé
        p["asset_value"] = calculate_asset_value(p["trucks"])

        # 0. Gestion des faillites (Vente d'actifs pour récupérer)
        if not p["active"] and not p.get("can_recover", True):
            p["history"].append("🚨 Entreprise liquidée. Aucune action possible.")
            continue
            
        if not p["active"] and p.get("can_recover", True):
            if "sell_trucks" in action:
                # Dans ce cas, 'sell_trucks' est une liste d'UUIDs à vendre (humain ou IA)
                for model_id, uuid_list in action["sell_trucks"].items():
                    if isinstance(uuid_list, list):
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
                continue # Ne pas traiter les actions du tour si l'entreprise est en faillite temporaire
            
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
            else:
                p["history"].append(f"R&D ({rd_type_chosen}) annulée : Fonds insuffisants.")
                rd_type_chosen = "Aucun"
        
        # C. Dépenses d'exploitation
        total_fixed_costs = FIXED_COSTS 
        total_variable_costs = 0
        
        # Coût événementiel
        cost_modifier = 1.0
        if current_event["type"] == "Carburant":
            if p["rd_investment_type"] == "Carburant":
                cost_modifier = current_event.get("cost_protection_covered", 1.0)
            else:
                cost_modifier = current_event.get("cost_increase", 1.0)
        
        # 1. Salaires
        payroll = p["employees"] * SALARY_PER_EMP
        total_fixed_costs += payroll
        
        # 2. Maintenance et assurance
        for truck in p["trucks"]:
            maintenance = truck.get("maintenance", 0) * cost_modifier
            insurance = INSURANCE_PER_TRUCK_BASE * cost_modifier
            total_fixed_costs += int(maintenance + insurance)
        
        # 3. Coûts d'emprunt et de R&D (fixes)
        interest_cost = int(p["loan"] * INTEREST_RATE_PER_TURN)
        total_fixed_costs += interest_cost
        p["loan"] += interest_cost # L'intérêt est ajouté au prêt
        p["loan_age"] += 1
        
        if rd_type_chosen != "Aucun":
            rd_config = R_D_TYPES.get(rd_type_chosen, {})
            rd_cost = rd_config.get("cost", 0)
            total_fixed_costs += rd_cost
            p["history"].append(f"Dépense R&D ({rd_type_chosen}): -{rd_cost:,} €".replace(",", " "))

        # 4. Coûts imprévus (événementiels)
        unforeseen_cost_mod = current_event.get("unforeseen_cost_mod", 1.0)
        unforeseen_costs = int(random.uniform(2000, 5000) * unforeseen_cost_mod)
        total_fixed_costs += unforeseen_costs
        p["history"].append(f"Coûts imprévus (réglementation/autre): -{unforeseen_costs:,} €".replace(",", " "))

        # 5. Achats/Ventes de Camions
        if "buy_trucks" in action:
            for model_id, qty in action["buy_trucks"].items():
                model = next(m for m in TRUCK_MODELS if m["id"] == model_id)
                cost = model["purchase_price"] * qty
                if p["money"] >= cost:
                    p["money"] -= cost
                    p["trucks"].extend([_new_truck(model) for _ in range(qty)])
                    p["history"].append(f"Achat de {qty}x {model_id}: -{cost:,} €".replace(",", " "))
                else:
                    p["history"].append(f"Achat de {qty}x {model_id} annulé : Fonds insuffisants.")

        if "sell_trucks" in action:
            for model_id, uuid_list in action["sell_trucks"].items():
                for truck_uuid_str in uuid_list:
                    truck_to_sell = next((t for t in p["trucks"] if str(t.get("uuid")) == truck_uuid_str), None)
                    if truck_to_sell:
                        p["trucks"].remove(truck_to_sell)
                        current_value = truck_to_sell["purchase_price"] * (1 - truck_to_sell["age"] * 0.10) 
                        resale = int(max(truck_to_sell["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                        p["money"] += resale
                        p["history"].append(f"Vente de {truck_to_sell['id']}: +{resale:,} €".replace(",", " "))


        # C. Gains et Réputation (Calcul des revenus)
        delivered_packages_capacity = allocation_capacity.get(p["name"], {t: 0 for t in COLIS_TYPES})
        total_revenue = 0
        new_reputation = p["reputation"]
        
        for t in COLIS_TYPES:
            capacity_units_delivered = delivered_packages_capacity.get(t, 0)
            packages_delivered = int(capacity_units_delivered / CAPACITY_UNITS_PER_COLIS.get(t, 1.0))
            
            revenue = packages_delivered * p["prices"].get(t, 0)
            
            total_revenue += revenue
            p["delivered_packages_total"][t] += packages_delivered
            total_variable_costs += packages_delivered * 5 # Coût variable par colis
        
        p["money"] += total_revenue
        
        # Application des dépenses
        final_expenses = total_fixed_costs + total_variable_costs
        p["money"] -= final_expenses
        
        p["income"] = total_revenue
        p["expenses"] = final_expenses
        
        p["history"].append(f"Revenus du Tour (Livraison) : +{total_revenue:,} €".replace(",", " "))
        p["history"].append(f"Dépenses Totales : -{final_expenses:,} €".replace(",", " "))
        
        # D. Mise à jour de la réputation
        # Réputation de base: 1% de l'argent gagné (max 0.1)
        rep_base_gain = min(0.1, total_revenue / 1000000)
        new_reputation += rep_base_gain
        
        # Impact de l'événement sur la réputation
        if current_event["type"] == "Cyber":
            if p["rd_investment_type"] == "CyberSécurité":
                penalty = current_event.get("rep_penalty", 1.0)
                new_reputation *= penalty
                p["history"].append("CyberSécurité R&D: Réduction de la pénalité de réputation.")
            else:
                penalty = current_event.get("rep_penalty_uncovered", 0.7)
                new_reputation *= penalty
                p["history"].append("CyberSécurité: Pénalité de réputation maximale appliquée.")
        
        # Clôture de la réputation
        p["reputation"] = max(0.5, min(2.5, new_reputation))
        p["history"].append(f"Réputation actuelle: {p['reputation']:.2f}")

        # E. Vieillissement des camions
        for truck in p["trucks"]:
            truck["age"] += 1
        
        # F. Vérification de faillite (dette excessive)
        asset_val = calculate_asset_value(p["trucks"])
        p["asset_value"] = asset_val
        
        if p["loan"] > 0 and p["loan"] / max(1, asset_val) >= FAILLITE_RATIO:
            p["active"] = False
            p["can_recover"] = True # Peut tenter de vendre des actifs pour récupérer
            p["history"].append("🚨 FAILLITE: Dettes excessives par rapport aux actifs. Vous avez une chance de vendre un camion.")
        elif p["loan"] > 0 and p["loan_age"] > MAX_LOAN_AGE_BEFORE_SEIZURE:
            p["active"] = False
            p["can_recover"] = False # Trop de tours sans remboursement, saisie et fin de partie
            p["history"].append("❌ SAISIE: Prêt impayé depuis trop longtemps. L'entreprise est liquidée.")

        # G. Mise à jour R&D Logistique
        if p["rd_investment_type"] == "Logistique":
            p["rd_boost_log"] = R_D_TYPES["Logistique"]["boost_value"]
        else:
            p["rd_boost_log"] = 0
            
        game_state["players"][i] = p
    
    # --- PHASE POST-TOUR ---
    game_state["turn"] += 1
    # Réinitialiser les actions des joueurs humains pour le prochain tour
    for p in game_state["players"]:
        if p["is_human"]:
            game_state["players_ready"][p["name"]] = False
            game_state["actions_this_turn"].pop(p["name"], None)
        
    # Sauvegarde de l'état du jeu pour la synchronisation
    save_game_state_to_db(game_state["game_id"], game_state)
    
    return game_state

# ---------------- SIMULATION D'INTERFACE STREAMLIT (DEMO) ----------------

# Cette section est ajoutée pour montrer comment un joueur rejoignant la partie 
# enverrait une demande de jointure, et comment le jeu démarre.

def show_join_game_ui():
    """Interface pour un joueur qui veut rejoindre une partie existante."""
    st.title("Rejoindre une Partie Existante")
    st.markdown("---")
    
    st.warning("Pour simuler l'envoi d'une demande de jointure, entrez l'ID de la partie et votre nom.")
    
    join_game_id = st.text_input("ID de la Partie à rejoindre (ex: GAME-1A2B3C)", key="join_game_id_input").strip().upper()
    join_player_name = st.text_input("Votre Nom/Nom d'Entreprise", key="join_player_name_input").strip()
    
    if st.button("Soumettre la Demande de Jointure", type="primary"):
        if join_game_id and join_player_name:
            submit_join_request(join_game_id, join_player_name)
        else:
            st.error("Veuillez remplir l'ID de la partie et votre nom.")

# Placeholder pour la page d'accueil/lancement
if 'game_state' not in st.session_state:
    st.session_state.current_page = 'home'
    st.session_state.my_name = None
    st.session_state.game_id = None
    st.session_state.is_host = False

if st.session_state.current_page == 'home':
    st.title("Jeu de Logistique Multijoueur")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Créer une Nouvelle Partie")
        host_name = st.text_input("Votre Nom d'Entreprise (Contrôleur)", key="create_host_name")
        num_ia = st.number_input("Nombre d'IA concurrentes", min_value=0, max_value=5, value=2, key="create_num_ia")
        participate = st.checkbox("Le contrôleur participe en tant que joueur", value=True)
        
        if st.button("Créer et Lancer Lobby", type="primary"):
            if host_name:
                initialize_game_state(host_name, num_ia, participate)
                st.session_state.current_page = 'lobby'
                st.session_state.is_host = True
                st.rerun()
            else:
                st.error("Veuillez entrer un nom d'hôte.")
                
    with col2:
        show_join_game_ui()
        
elif st.session_state.current_page == 'lobby':
    if st.session_state.is_host:
        show_lobby_host(st.session_state.game_id, st.session_state.my_name)
    else:
        st.info(f"Connecté à la partie {st.session_state.game_id}. En attente de l'approbation du contrôleur ({st.session_state.game_state.get('host_name', 'Hôte inconnu')}).")
        # Un joueur rejoignant la partie aurait ici une UI de 'Jeu en Attente'
        st.markdown("---")
        if st.button("Actualiser le Statut du Jeu", type="secondary"):
             # Forcer la synchronisation pour voir si l'hôte a accepté l'état du jeu (même si l'hôte ne nous connaît pas encore comme joueur)
             sync_game_state(st.session_state.game_id)
