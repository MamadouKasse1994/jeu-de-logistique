import streamlit as st
import pandas as pd
import datetime
import os
import uuid
import random
from copy import deepcopy
import numpy as np # Nécessaire pour poisson_market

# --- 1. CONSTANTES GLOBALES ET PARAMÈTRES DE JEU ---

# Environnement/API (Simulé)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "http://fake-api.com")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "fake_key")

# Paramètres du Jeu
COLIS_TYPES = ["Standard", "Express", "Fragile"] # Types de colis
BASE_PRICES = {"Standard": 150, "Express": 300, "Fragile": 400} # Prix de base pour l'IA
CAPACITY_UNITS_PER_COLIS = {"Standard": 1.0, "Express": 1.5, "Fragile": 2.0} # Taille relative
TOTAL_CLIENTS = 100 # Nombre de clients simulés par tour
BASE_DEMANDS = {"Standard": 2000, "Express": 1000, "Fragile": 500} # Demande de capacité de base

# Paramètres des Joueurs/Entreprises
INITIAL_MONEY = 100000
FIXED_COSTS = 15000 # Loyer, énergie, etc.
SALARY_PER_EMP = 5000
INDEMNITY_PER_EMP = 15000 # Coût de licenciement
INSURANCE_PER_TRUCK_BASE = 500
MAX_LOAN_AGE_BEFORE_SEIZURE = 5
INTEREST_RATE_PER_TURN = 0.05
MIN_LOAN_PAYMENT_RATIO = 0.10
MIN_TRUCK_RESALE_RATIO = 0.3 # Valeur de revente minimale

# Paramètres des Camions
PERF_LOSS_PER_AGE = 0.05
REFERENCE_SPEED = 60 # Vitesse de référence pour le calcul des voyages
MAX_TRIPS = 2 # Max de voyages par camion
TRUCK_MODELS = [
    {"id": "Eco", "capacity": 1000, "speed": 60, "price": 40000, "maintenance": 1500},
    {"id": "Standard", "capacity": 1500, "speed": 70, "price": 60000, "maintenance": 2000},
    {"id": "Heavy", "capacity": 2500, "speed": 50, "price": 90000, "maintenance": 3000},
]

# R&D
R_D_TYPES = {
    "Logistique": {"cost": 30000, "defense_type": "Logistique", "boost_value": 0.05},
    "Carburant": {"cost": 25000, "defense_type": "Carburant"},
    "CyberSécurité": {"cost": 35000, "defense_type": "CyberSécurité"},
}

# Événements (Liste simplifiée pour l'exécution)
EVENT_LIST = [
    {"name": "Hausse du Carburant", "text": "Les coûts de transport augmentent.", "type": "Carburant", "cost_increase": 1.25, "cost_protection_covered": 1.05, "rep_penalty_uncovered": 1.0, "rep_penalty": 1.0},
    {"name": "Cyberattaque", "text": "Les systèmes sont ralentis.", "type": "CyberSécurité", "cost_increase": 1.0, "rep_penalty_uncovered": 0.8, "rep_penalty": 1.0},
    {"name": "Grève des Dockers", "text": "Retards et pénalités de réputation.", "type": "Social", "cost_increase": 1.0, "rep_penalty_uncovered": 0.85, "rep_penalty": 1.0},
]


# ----------------------------------------------------------------------
#                             2. FONCTIONS ENTITÉS ET DB (STUBS)
# ----------------------------------------------------------------------

def generate_player_names(count, existing_names):
    """Génère des noms d'IA uniques."""
    ia_names = [f"IA {i+1}" for i in range(count)]
    # Assurez-vous qu'ils sont uniques (simplifié)
    return [name for name in ia_names if name not in existing_names]

def create_player_entity(name, is_human=True):
    """Crée l'entité de base pour un joueur/une IA."""
    
    # Initialisation des camions
    initial_truck = deepcopy(TRUCK_MODELS[0])
    initial_truck["age"] = 0
    initial_truck["uuid"] = uuid.uuid4()
    initial_truck["purchase_price"] = initial_truck["price"]
    
    entity = {
        "name": name,
        "is_human": is_human,
        "active": True,
        "money": INITIAL_MONEY,
        "reputation": 1.0,
        "loan": 0,
        "loan_age": 0,
        "trucks": [initial_truck],
        "total_capacity": calculate_player_capacity({"trucks": [initial_truck], "rd_boost_log": 0}), # Calculé plus tard
        "employees": 5,
        "rd_level": 0,
        "rd_investment_type": "Aucun",
        "rd_boost_log": 0.0, # Boost de capacité par R&D Logistique
        "prices": deepcopy(BASE_PRICES),
        "income": 0,
        "expenses": 0,
        "asset_value": initial_truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO,
        "history": [f"Entreprise {name} créée."],
        "can_recover": False,
        "delivered_packages_total": {t: 0 for t in COLIS_TYPES},
    }
    entity["total_capacity"] = calculate_player_capacity(entity)
    return entity

def load_game_state_from_db(game_id):
    """(STUB DB) Charge l'état du jeu depuis la DB."""
    # En production, ce serait un appel API à Supabase (GET)
    return st.session_state.get('game_state')

def save_game_state_to_db(game_id, game_state):
    """(STUB DB) Sauvegarde l'état du jeu dans la DB."""
    # En production, ce serait un appel API à Supabase (UPSERT)
    st.session_state.game_state = game_state

def load_game_chat(game_id):
    """(STUB DB) Charge les messages de chat pour une partie."""
    # En production, ce serait un appel à une table de chat.
    return st.session_state.game_state.get('chat_messages', [])

def update_game_chat(game_id, sender, message):
    """(STUB DB) Ajoute un message au chat de la partie."""
    # En production, ce serait un appel API à Supabase (INSERT)
    new_message = {
        "sender": sender,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat()
    }
    chat_messages = st.session_state.game_state.get('chat_messages', [])
    chat_messages.append(new_message)
    st.session_state.game_state['chat_messages'] = chat_messages
    save_game_state_to_db(game_id, st.session_state.game_state)

def submit_join_request(game_id, player_name):
    """(STUB DB) Envoie une demande de jointure à l'hôte."""
    # Charger l'état (simulé ici par load_game_state_from_db)
    game_state = load_game_state_from_db(game_id)
    if game_state and game_state['game_status'] == 'lobby':
        if player_name not in game_state.get('pending_players', []) and player_name not in [p['name'] for p in game_state['players']]:
            if 'pending_players' not in game_state: game_state['pending_players'] = []
            game_state['pending_players'].append(player_name)
            save_game_state_to_db(game_id, game_state)
            return True
    return False

def sync_game_state(game_id):
    """Synchronise l'état du jeu avec la base de données."""
    # En production, ceci doit interroger la DB via API
    synced_state = load_game_state_from_db(game_id)
    if synced_state:
        st.session_state.game_state = synced_state
        st.session_state.game_id = game_id
        # Logique de changement d'écran après sync
        current_status = synced_state.get('game_status', 'lobby')
        if current_status == 'in_progress' and st.session_state.screen != 'game':
             st.session_state.screen = 'game'
        elif current_status == 'lobby' and st.session_state.my_name == synced_state['host_name']:
             st.session_state.screen = 'lobby'
        st.rerun()
    else:
        st.warning("Impossible de synchroniser. Vérifiez l'ID de la partie.")

# ----------------------------------------------------------------------
#                             3. FONCTIONS DE JEU (Inchangées/Corrigées)
# ----------------------------------------------------------------------

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

def calculate_player_capacity(player_data):
    """Calcule la capacité totale de livraison effective d'un joueur."""
    total_capacity = 0
    log_rd_boost = player_data.get("rd_boost_log", 0)
    
    for truck in player_data["trucks"]:
        # Correction pour vérifier l'existence des clés de base
        if not isinstance(truck, dict) or 'capacity' not in truck or 'age' not in truck: continue 
        
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
        if not isinstance(truck, dict) or 'purchase_price' not in truck or 'age' not in truck: continue
        
        current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
        resale = max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value)
        total_value += resale
    return int(total_value)

def poisson_market(base, trend=1.0):
    """Génère la demande de base du marché selon une distribution de Poisson."""
    # numpy.random.poisson est utilisé
    return int(np.random.poisson(max(0, base * trend)))

def generate_client_orders(game_state):
    """Génère la demande totale du marché (backlog + nouvelle demande + tendance/événements)."""
    package_orders = {t: 0 for t in COLIS_TYPES}
    
    # 1. Ajout du Backlog
    for t in COLIS_TYPES:
        package_orders[t] += game_state["backlog_packages"].get(t, 0)
        
    # 2. Ajout des nouvelles commandes
    for _ in range(TOTAL_CLIENTS * 2): 
        # Simuler des commandes de différentes tailles
        types_chosen = random.choices(COLIS_TYPES, k=random.randint(1, 3))
        for t in types_chosen:
            package_orders[t] += random.randint(1, 5)

    # 3. Ajout de la demande de base (Poisson/Tendance)
    for t in COLIS_TYPES:
        package_orders[t] += poisson_market(BASE_DEMANDS.get(t, 0), game_state["market_trend"]) 
        
    # 4. Impact de l'événement
    if "market_effect" in game_state["current_event"]:
        for t in package_orders:
            package_orders[t] = int(package_orders[t] * game_state["current_event"]["market_effect"])

    # Calcul de la capacité requise (en unités de capacité)
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
    # Plus le score est bas (bon prix/haute réputation), plus le poids est élevé
    attractiveness_weight = 1.0 / (total_score * total_score) 
    return attractiveness_weight, player_exec_capacity

def distribute_clients(market_capacity_demand, players, game_state):
    """Alloue la demande du marché aux joueurs en fonction de leur capacité et de leur score."""
    allocation_capacity = {p["name"]: {t: 0 for t in COLIS_TYPES} for p in players}
    current_package_backlog = {t: 0 for t in COLIS_TYPES}
    active_players = [p for p in players if p["active"]]
    
    # Préparation des données pour la distribution
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
        unit_size = max(1, qty_capacity_remaining // 4) # Distribution par paquets
        
        # Logique de distribution itérative (style Roulette Wheel Selection)
        while qty_capacity_remaining > 0:
            scores_and_weights = []
            for p_name, data in player_data.items():
                p = data["player"]
                cap_remaining_global = data["max_capacity"] - data["current_allocation_total"]
                
                if cap_remaining_global > 0:
                    scores_and_weights.append({
                        "player": p, 
                        "weight": data["scores"].get(t, 0) * (cap_remaining_global / p["total_capacity"]), # Poids pondéré par la capacité restante
                        "cap_remaining_global": cap_remaining_global
                    })

            total_market_weight = sum(item["weight"] for item in scores_and_weights)
            
            if not scores_and_weights or total_market_weight <= 0.0001:
                # Plus de joueurs actifs ou plus de poids significatifs
                break
            
            weights = [item["weight"] for item in scores_and_weights]
            players_chosen = [item["player"] for item in scores_and_weights]
            
            chosen_player = random.choices(players_chosen, weights=weights, k=1)[0]
            
            p_name = chosen_player["name"]
            data = player_data[p_name]
            cap_remaining = data["max_capacity"] - data["current_allocation_total"]
            
            capacity_to_distribute = min(unit_size, qty_capacity_remaining) 
            deliverable_capacity = min(capacity_to_distribute, cap_remaining)
            
            if deliverable_capacity > 0:
                allocation_capacity[p_name][t] += deliverable_capacity
                qty_capacity_remaining -= deliverable_capacity
                player_data[p_name]["current_allocation_total"] += deliverable_capacity
            
        # Mise à jour du Backlog
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

def get_ia_actions(player_data):
    """Détermine les actions d'un joueur IA basé sur une stratégie simple."""
    actions = {}
    
    new_prices = deepcopy(player_data["prices"])
    
    # 1. Logique de Faillite (vente d'actifs)
    if not player_data["active"]:
        actions["sell_trucks"] = {}
        # Vendre le camion le plus cher/vieux (simplifié)
        trucks_sorted = sorted(player_data["trucks"], key=lambda t: t.get("purchase_price", 0) * (1 - t.get("age", 0)*0.1), reverse=True)
        if trucks_sorted and len(player_data["trucks"]) > 1: # Ne pas vendre le dernier
            model_id = trucks_sorted[0]["id"]
            truck_uuid = str(trucks_sorted[0]["uuid"])
            actions["sell_trucks"][model_id] = [truck_uuid]
        return actions 

    # 2. Gestion des prix
    price_mod = 0.98 if player_data["reputation"] < 1.0 else 1.02
    for t in COLIS_TYPES:
        # L'IA Logistique peut baisser les prix en cas de bon RD
        price_rd_factor = (1 - (player_data["rd_boost_log"] / 2))
        new_prices[t] = max(50, int(BASE_PRICES[t] * price_mod * price_rd_factor))
    actions["prices"] = new_prices

    # 3. Investissement et Capacité
    money_threshold = 40000 + (player_data["turn"] * 5000)
    current_capacity = player_data["total_capacity"]
    
    # R&D
    if player_data["money"] > money_threshold and player_data["rd_boost_log"] < 0.2:
        actions["rd_type"] = "Logistique"
    elif player_data["money"] > money_threshold * 1.5 and player_data["rd_investment_type"] == "Aucun":
        actions["rd_type"] = random.choice(["Carburant", "CyberSécurité"])
    else:
        actions["rd_type"] = "Aucun"
        
    # Achat de Camions
    actions["buy_trucks"] = {}
    if current_capacity < 1500 and player_data["money"] > 60000:
        model_to_buy = TRUCK_MODELS[0]
        actions["buy_trucks"][model_to_buy["id"]] = 1
    elif current_capacity < 3000 and player_data["money"] > 100000 and random.random() < 0.5:
        model_to_buy = TRUCK_MODELS[1]
        actions["buy_trucks"][model_to_buy["id"]] = 1
        
    # 4. Gestion des employés
    target_employees = max(5, int(current_capacity / 1000))
    emp_delta = target_employees - player_data["employees"]
    if abs(emp_delta) > 1 and random.random() < 0.3:
        actions["emp_delta"] = 1 if emp_delta > 0 else -1
    
    # 5. Publicité (occasionnelle)
    if player_data["reputation"] < 1.5 and player_data["money"] > 25000 and random.random() < 0.2:
        actions["pub_type"] = "Nationale" # Action non implémentée, mais dans la logique IA
        
    # 6. Remboursement Prêt (si prêt très élevé)
    if player_data["loan"] > 0 and player_data["loan"] > player_data["money"] * 2 and random.random() < 0.4:
        actions["pay_extra_loan"] = player_data["money"] * 0.5
        
    return actions

def apply_truck_action(player_entity, actions):
    """Applique les achats et ventes de camions."""
    
    # A. Vente
    # actions["sell_trucks"] = {"Eco": ["uuid1", "uuid2"], ...}
    for model_id, truck_uuids in actions.get("sell_trucks", {}).items():
        if not truck_uuids: continue
        
        trucks_to_keep = []
        sold_count = 0
        
        # Crée un ensemble de tous les UUIDs à vendre pour ce modèle
        uuids_to_sell = set(truck_uuids)
        
        for truck in player_entity["trucks"]:
            if str(truck.get("uuid")) in uuids_to_sell and len(player_entity["trucks"]) - sold_count > 1:
                # Calculer le prix de vente
                current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
                sale_price = max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value)
                
                player_entity["money"] += sale_price
                player_entity["income"] += sale_price # Compté comme revenu
                player_entity["history"].append(f"Vente du camion {truck['id']} (Âge {truck['age']}) pour {sale_price:,.0f}€.")
                sold_count += 1
                uuids_to_sell.remove(str(truck.get("uuid"))) # Retirer de l'ensemble à vendre
            else:
                trucks_to_keep.append(truck)
        
        player_entity["trucks"] = trucks_to_keep
        
        if sold_count == 0 and truck_uuids:
             player_entity["history"].append(f"Avertissement: Tentative de vente non effectuée (dernier camion ou UUID non trouvé).")


    # B. Achat
    # actions["buy_trucks"] = {"Standard": 1, "Heavy": 2}
    for model_id, quantity in actions.get("buy_trucks", {}).items():
        if quantity <= 0: continue
        
        model_info = next((m for m in TRUCK_MODELS if m["id"] == model_id), None)
        if not model_info: continue
        
        cost = model_info["price"]
        
        for _ in range(quantity):
            if player_entity["money"] >= cost:
                new_truck = deepcopy(model_info)
                new_truck["age"] = 0
                new_truck["uuid"] = uuid.uuid4()
                new_truck["purchase_price"] = cost
                
                player_entity["money"] -= cost
                player_entity["expenses"] += cost # Compté comme dépense
                player_entity["trucks"].append(new_truck)
                player_entity["history"].append(f"Achat du camion {model_id} pour -{cost:,.0f}€.")
            else:
                player_entity["history"].append(f"Fonds insuffisants pour acheter {model_id}.")
                break
                
    # Recalculer la capacité totale
    player_entity["total_capacity"] = calculate_player_capacity(player_entity)
    return player_entity


def simulate_turn_streamlit(game_state, actions_dict):
    """Exécute un tour de simulation."""
    
    # 0. Déclenchement de l'événement et info du tour
    trigger_random_event(game_state)
    current_event = game_state["current_event"]
    event_info = f"🌪️ Événement du Tour: **{current_event['name']}** - {current_event['text']}"
    
    # 1. Actions IA (Capacité pour l'IA et actions IA)
    for p in game_state["players"]:
        if not p["is_human"]:
            # Recalculer la capacité pour le tour (inclut le boost RD)
            p_cap = calculate_player_capacity(p)
            p["total_capacity"] = p_cap
            # Déterminer les actions de l'IA
            ia_action = get_ia_actions(p)
            actions_dict[p["name"]] = ia_action
    
    # 2. Génération de la demande et distribution du marché
    market_capacity_demand = generate_client_orders(game_state) 
    allocation_capacity = distribute_clients(market_capacity_demand, game_state["players"], game_state)

    # --- PHASE D'APPLICATION DES ACTIONS ET CALCUL DES FINANCES ---
    for i, p in enumerate(game_state["players"]):
        
        if not p["active"]: 
            p["history"].append(f"Entreprise inactive (Faillite).")
            continue
            
        p["history"] = [event_info] # Réinitialiser l'historique du tour
        actions = actions_dict.get(p["name"], {})
        
        # A. Application des actions des Joueurs (Camions, Prix, R&D, Empl.)
        
        # A.1. Mise à jour des Prix (faite lors de la soumission de l'action)
        new_prices_action = actions.get("prices")
        if new_prices_action:
            p["prices"] = new_prices_action
            
        # A.2. Achat/Vente de camions
        p = apply_truck_action(p, actions)
        
        # A.3. R&D (inclut le coût et le boost)
        rd_type = actions.get("rd_type")
        if rd_type != "Aucun" and rd_type in R_D_TYPES:
            rd_cost = R_D_TYPES[rd_type]["cost"]
            if p["money"] >= rd_cost:
                p["expenses"] += rd_cost
                p["rd_investment_type"] = rd_type
                if rd_type == "Logistique":
                    p["rd_boost_log"] += R_D_TYPES[rd_type]["boost_value"]
                    p["history"].append(f"Investissement R&D Logistique. Boost actuel: {p['rd_boost_log'] * 100:.1f}%.")
                p["history"].append(f"Investissement R&D ({rd_type}): -{rd_cost:,.0f}€")
            else:
                p["history"].append("Échec R&D: Fonds insuffisants.")
                p["rd_investment_type"] = "Aucun"
        else:
            p["rd_investment_type"] = "Aucun"
            
        # A.4. Employés
        emp_delta = actions.get("emp_delta", 0)
        if emp_delta != 0:
            if emp_delta > 0:
                p["employees"] += emp_delta
                p["history"].append(f"Embauche de {emp_delta} employé(s).")
            elif emp_delta < 0 and p["employees"] + emp_delta >= 1: # Toujours au moins 1 employé
                release_cost = abs(emp_delta) * INDEMNITY_PER_EMP
                p["employees"] += emp_delta
                p["expenses"] += release_cost
                p["history"].append(f"Licenciement de {-emp_delta} employé(s). Coût: -{release_cost:,.0f}€")
            elif emp_delta < 0:
                p["history"].append("Impossible de licencier: nécessite au moins 1 employé.")

        # B. Revenues et Livraisons
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
            
        p["history"].append(f"Revenus du marché: +{p['income']:,.0f}€ (Capacité totale: {p['total_capacity']:,.0f} u).")
        

        # C. Dépenses et Coûts Fixes (Déjà calculé pour R&D/Camions/Employés)
        
        # C.1. Coûts fixes et salaires
        event_cost_mod = current_event.get("cost_increase", 1.0)
        
        if current_event["type"] == "Carburant" and p["rd_investment_type"] == "Carburant":
            event_cost_mod = current_event.get("cost_protection_covered", 1.0)

        salaries = p["employees"] * SALARY_PER_EMP
        fixed_costs = FIXED_COSTS * event_cost_mod
        p["expenses"] += salaries + fixed_costs
        p["history"].append(f"Frais fixes et Salaires: -{fixed_costs + salaries:,.0f}€ (Facteur coût: {event_cost_mod:.2f})")

        # C.2. Entretien des camions et assurance
        maintenance_cost = sum(t.get("maintenance", 0) for t in p["trucks"])
        insurance_cost = len(p["trucks"]) * INSURANCE_PER_TRUCK_BASE
        p["expenses"] += maintenance_cost + insurance_cost
        p["history"].append(f"Entretien et Assurances: -{maintenance_cost + insurance_cost:,.0f}€")

        # C.3. Remboursement de prêt et intérêts
        if p["loan"] > 0:
            interest = p["loan"] * INTEREST_RATE_PER_TURN
            min_payment = p["loan"] * MIN_LOAN_PAYMENT_RATIO
            
            # Application du paiement minimum
            total_payment = min_payment + interest
            
            if p["money"] >= total_payment:
                p["loan"] -= min_payment
                p["expenses"] += total_payment
                p["money"] -= total_payment # Soustraire ici car c'est une action de flux de trésorerie
                p["loan_age"] = 0
                p["history"].append(f"Prêt (Intérêts/Principal): -{total_payment:,.0f}€ (Prêt restant: {p['loan']:,.0f}€)")
            else:
                p["loan"] += interest # Les intérêts s'accumulent
                p["loan_age"] += 1
                p["history"].append(f"ATTENTION: Paiement de prêt non effectué. Intérêts cumulés. Âge du prêt: {p['loan_age']}.")
        
        # D. Vieillissement des Actifs
        for truck in p["trucks"]:
            truck["age"] += 1
            
        # E. Réputation et Événements
        
        # Pénalité de réputation (si événement non couvert)
        if current_event.get("rep_penalty_uncovered") and p["rd_investment_type"] != current_event["type"]:
            p["reputation"] *= current_event["rep_penalty_uncovered"]
            p["history"].append(f"Impact négatif de l'événement {current_event['name']} (non couvert) sur la réputation.")
        elif current_event.get("rep_penalty"):
            p["reputation"] *= current_event["rep_penalty"]

        # Régression vers la moyenne
        p["reputation"] = p["reputation"] * 0.95 + 0.05 * 1.0
        p["reputation"] = max(0.5, min(2.0, p["reputation"]))


        # F. Mouvement de Trésorerie Final (Revenus - Dépenses non déduites ci-dessus)
        p["money"] += p["income"] - p["expenses"]
        
        # G. Fin de Tour et Faillite
        p["asset_value"] = calculate_asset_value(p["trucks"])
        
        if p["money"] < 0 and p["active"]:
            # Logique de faillite/danger
            if p["loan"] > 0 and p["loan_age"] >= MAX_LOAN_AGE_BEFORE_SEIZURE:
                p["active"] = False
                p["history"].append("FAILLITE (Prêt non remboursé et au-delà de l'âge limite)!")
            elif abs(p["money"]) > 1.5 * p["asset_value"]:
                p["active"] = False
                p["history"].append("FAILLITE (Dette trop importante par rapport aux actifs)!")
            elif abs(p["money"]) > p["asset_value"] * 0.8:
                p["history"].append("ATTENTION: Votre entreprise est en **danger de faillite**.")
            else:
                 p["history"].append(f"Trésorerie Négative: -{abs(p['money']):,.0f}€. Action de survie requise.")
        
        game_state["players"][i] = p
        
    game_state["turn"] += 1
    game_state["players_ready"] = {p["name"]: False for p in game_state["players"] if p["is_human"]}
    game_state["actions_this_turn"] = {}
    return game_state

# ----------------------------------------------------------------------
#                             4. FONCTIONS D'INTERFACE (Inchangées)
# ----------------------------------------------------------------------

# ... (Les fonctions show_sidebar, render_chat, show_setup_screen, show_pending_screen, show_lobby_host, show_in_progress_game, et main sont inchangées et complètes) ...

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
            if submit_join_request(join_game_id, player_name): # <--- C'est ici
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

    if not my_player_entity or not my_player_entity['active']:
        st.error("Votre entreprise n'est plus active dans cette partie (Faillite ou observateur).")
        return

    # AFFICHER L'ÉTAT ACTUEL DE LA PARTIE
    col_money, col_rep, col_loan, col_cap = st.columns(4)
    # Le delta est basé sur le résultat du tour précédent
    delta_val = my_player_entity['income'] - my_player_entity['expenses'] 
    col_money.metric("Trésorerie", f"{my_player_entity['money']:,.0f} €", delta=delta_val if delta_val != 0 else None)
    col_rep.metric("Réputation", f"{my_player_entity['reputation']:.2f}")
    col_loan.metric("Prêt Restant", f"{my_player_entity['loan']:,.0f} €")
    col_cap.metric("Capacité Effective", f"{my_player_entity['total_capacity']:,.0f} u")

    st.markdown("---")
    
    st.subheader(f"Actions de l'Entreprise : {my_name}")
    
    # Historique Récent
    last_history = my_player_entity['history'][-5:] 
    st.markdown("#### Historique Récent")
    for msg in reversed(last_history):
        st.text(f"- {msg}")
    
    st.markdown("---")
    
    # Vérification si les actions ont déjà été soumises
    is_ready = st.session_state.is_ready if 'is_ready' in st.session_state else game_state['players_ready'].get(my_name, False)
    
    if is_ready:
        st.success("✅ Vos actions pour ce tour ont été soumises. En attente des autres joueurs...")
        not_ready_count = len([name for name, ready in game_state['players_ready'].items() if not ready and name != my_name])
        st.info(f"Joueurs non prêts : {not_ready_count}")
        
        # LOGIQUE HÔTE : Passage au tour suivant
        if my_name == game_state['host_name'] and all(game_state['players_ready'].values()):
            if st.button("▶️ Lancer le Tour Suivant (Host)", type="primary"):
                st.toast("Simulation du tour en cours...")
                new_game_state = simulate_turn_streamlit(game_state, game_state['actions_this_turn'])
                save_game_state_to_db(game_state['game_id'], new_game_state)
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
    col_buy, col_sell, col_info = st.columns(3)
    
    # Achat
    buy_actions = {}
    truck_model_names = [m["id"] for m in TRUCK_MODELS]
    model_to_buy = col_buy.selectbox("Modèle à acheter", ["Aucun"] + truck_model_names, key="truck_buy_model")
    num_to_buy = 0
    if model_to_buy != "Aucun":
        model_info = next(m for m in TRUCK_MODELS if m["id"] == model_to_buy)
        col_info.info(f"Coût Achat: {model_info['price']:,.0f} €/u")
        num_to_buy = col_buy.number_input(f"Quantité à acheter", min_value=0, max_value=5, value=0, key="truck_buy_qty")
        if num_to_buy > 0:
            buy_actions[model_to_buy] = num_to_buy
            
    # Vente
    truck_options = {str(t["uuid"]): f"{t['id']} - Âge {t['age']} - Revente ~{calculate_asset_value([t]):,.0f} €" for t in my_player_entity["trucks"]}
    
    if len(my_player_entity["trucks"]) > 1:
        trucks_to_sell_uuids = col_sell.multiselect(
            f"Camions à vendre (min. 1 à garder)", 
            options=list(truck_options.keys()), 
            format_func=lambda x: truck_options[x],
            key="truck_sell_uuids"
        )
        # Convertir les UUIDs en action formatée par modèle (nécessaire pour la fonction apply_truck_action)
        sell_actions = {}
        for uuid_sell in trucks_to_sell_uuids:
            truck_entity = next(t for t in my_player_entity["trucks"] if str(t["uuid"]) == uuid_sell)
            if truck_entity:
                model = truck_entity["id"]
                sell_actions.setdefault(model, []).append(uuid_sell)
    else:
        col_sell.info("Vous devez garder au moins 1 camion.")
        sell_actions = {}
        
    col_info.info(f"Flotte Actuelle: {len(my_player_entity['trucks'])} camions")


    # 3. R&D
    st.markdown("#### 3. Investissement R&D")
    rd_choices_display = [f"{name} ({data['cost']} €)" for name, data in R_D_TYPES.items()]
    rd_choice_selected = st.selectbox(
        "Choisir l'investissement R&D", 
        ["Aucun"] + rd_choices_display, 
        key="rd_choice"
    )
    rd_choice = rd_choice_selected.split(" ")[0] if rd_choice_selected != "Aucun" else "Aucun"
    
    # 4. Gestion des Employés
    st.markdown("#### 4. Gestion des Employés")
    emp_delta_input = st.number_input(
        f"Changement d'employés (Actuel: {my_player_entity['employees']})", 
        min_value=-my_player_entity['employees'] + 1, # Toujours garder 1
        max_value=10, 
        value=0, 
        step=1, 
        key="emp_delta_input"
    )
    if emp_delta_input < 0:
        st.caption(f"Coût de licenciement: {-emp_delta_input * INDEMNITY_PER_EMP:,.0f} €")
        
    # 5. Finalisation et Soumission
    st.markdown("---")
    
    # Vérification des fonds pour l'achat de camions + R&D + Licenciement
    total_truck_cost = sum(
        next(m for m in TRUCK_MODELS if m["id"] == model)["price"] * qty
        for model, qty in buy_actions.items()
    )
    rd_cost = R_D_TYPES.get(rd_choice, {}).get("cost", 0)
    release_cost = abs(emp_delta_input) * INDEMNITY_PER_EMP if emp_delta_input < 0 else 0
    total_cost_to_check = total_truck_cost + rd_cost + release_cost
    
    if my_player_entity['money'] < total_cost_to_check:
        st.error(f"Fonds insuffisants pour ces actions ({total_cost_to_check:,.0f} € requis). Ajustez les achats/R&D/licenciements.")
        submit_disabled = True
    else:
        submit_disabled = False

    if st.button("Soumettre les Actions du Tour", type="primary", use_container_width=True, disabled=submit_disabled):
        
        current_actions = {
            "prices": new_prices,
            "rd_type": rd_choice,
            "buy_trucks": buy_actions,
            "sell_trucks": sell_actions,
            "emp_delta": emp_delta_input
        }
        
        # Mettre à jour l'état de l'entreprise localement pour la DB (les prix sont nécessaires pour le calcul du marché)
        player_index = next((i for i, p in enumerate(game_state["players"]) if p["name"] == my_name), -1)
        if player_index != -1:
            game_state["players"][player_index]["prices"] = new_prices 
        
        game_state['actions_this_turn'][my_name] = current_actions
        game_state['players_ready'][my_name] = True
        st.session_state.is_ready = True 
        
        save_game_state_to_db(game_id, game_state)
        st.toast("Actions soumises! En attente des autres joueurs.")
        st.rerun()


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
    st.set_page_config(layout="wide")
    main()