#%%
import numpy as np
path = '/Users/filippomerlo/Desktop/Datasets/numberbatch-en-19.08.txt'

word2vec = {}
with open(path, 'r') as file:
    next(file)  # skip the first line
    for _ in range(516782):
        line = file.readline().strip()
        if not line:  # Stop if the line is empty (end of file)
            break
        parts = line.split(' ')
        word2vec[parts[0]] = np.array([float(x) for x in parts[1:]])


#%%
import pandas as pd
import os

data_folder_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/THINGS'
# Path for object scene similarity metrics matices and norms
things_plus_size_mean_path = os.path.join(data_folder_path,'THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv')
things_plus_typicality_mean_path = os.path.join(data_folder_path,'THINGSplus/Metadata/Concept-specific/typicality_meanRatings.tsv')
things_plus_categories_path = os.path.join(data_folder_path,'THINGSplus/Metadata/Concept-specific/category53_wideFormat.tsv')

# IMPORT DATA
# Object scene similarity metrics matices and norms 
things_plus_size_mean_matrix = pd.read_csv(things_plus_size_mean_path, sep='\t', engine='python', encoding='utf-8')
things_plus_typicality_mean_matrix = pd.read_csv(things_plus_typicality_mean_path, sep='\t', engine='python', encoding='utf-8')
things_plus_categories = pd.read_csv(things_plus_categories_path, sep='\t', engine='python', encoding='utf-8')

id2context = {}
for i, row in things_plus_size_mean_matrix.iterrows():
    id2context[row['uniqueID']] = row['WordContext']
id2context
#%%
# NAMES AND CATEGORIES
# Scene Names
# categories of the SUN397 dataset
sun_scene_cat = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'apartment_building/outdoor', 'apse/indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival_gate/outdoor', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium', 'auto_factory', 'badlands', 'badminton_court/indoor', 'baggage_claim', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'basketball_court/outdoor', 'bathroom', 'batters_box', 'bayou', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'bistro/indoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden', 'bow_window/indoor', 'bow_window/outdoor', 'bowling_alley', 'boxing_ring', 'brewery/indoor', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cabin/outdoor', 'cafeteria', 'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior/backseat', 'car_interior/frontseat', 'carrousel', 'casino/indoor', 'castle', 'catacomb', 'cathedral/indoor', 'cathedral/outdoor', 'cavern/indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'chicken_coop/indoor', 'chicken_coop/outdoor', 'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'cloister/indoor', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'control_tower/outdoor', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'covered_bridge/exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle/office', 'dam', 'delicatessen', 'dentists_office', 'desert/sand', 'desert/vegetation', 'diner/indoor', 'diner/outdoor', 'dinette/home', 'dinette/vehicle', 'dining_car', 'dining_room', 'discotheque', 'dock', 'doorway/outdoor', 'dorm_room', 'driveway', 'driving_range/outdoor', 'drugstore', 'electrical_substation', 'elevator/door', 'elevator/interior', 'elevator_shaft', 'engine_room', 'escalator/indoor', 'excavation', 'factory/indoor', 'fairway', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'fire_escape', 'fire_station', 'firing_range/indoor', 'fishpond', 'florist_shop/indoor', 'food_court', 'forest/broadleaf', 'forest/needleleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'garage/indoor', 'garbage_dump', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'golf_course', 'greenhouse/indoor', 'greenhouse/outdoor', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub/outdoor', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail/indoor', 'jail_cell', 'jewelry_shop', 'kasbah', 'kennel/indoor', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'labyrinth/outdoor', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'library/indoor', 'library/outdoor', 'lido_deck/outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'moat/water', 'monastery/outdoor', 'mosque/indoor', 'mosque/outdoor', 'motel', 'mountain', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'music_store', 'music_studio', 'nuclear_power_plant/outdoor', 'nursery', 'oast_house', 'observatory/outdoor', 'ocean', 'office', 'office_building', 'oil_refinery/outdoor', 'oilrig', 'operating_room', 'orchard', 'outhouse/outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor', 'parking_garage/outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pilothouse/indoor', 'planetarium/outdoor', 'playground', 'playroom', 'plaza', 'podium/indoor', 'podium/outdoor', 'pond', 'poolroom/establishment', 'poolroom/home', 'power_plant/outdoor', 'promenade_deck', 'pub/indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'stadium/baseball', 'stadium/football', 'stage/indoor', 'staircase', 'street', 'subway_interior', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/indoor', 'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'temple/south_asia', 'tennis_court/indoor', 'tennis_court/outdoor', 'tent/outdoor', 'theater/indoor_procenium', 'theater/indoor_seats', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'track/outdoor', 'train_railway', 'train_station/platform', 'tree_farm', 'tree_house', 'trench', 'underwater/coral_reef', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court/indoor', 'volleyball_court/outdoor', 'waiting_room', 'warehouse/indoor', 'water_tower', 'waterfall/block', 'waterfall/fan', 'waterfall/plunge', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'wine_cellar/barrel_storage', 'wine_cellar/bottle_storage', 'wrestling_ring/indoor', 'yard', 'youth_hostel']
# 25
sen_scene_to_keep = ['kitchen','living_room','home_office','bathroom','restaurant','bedroom','delicatessen','bakery/shop','pantry','coffee_shop','cubicle/office','banquet_hall','market/outdoor','dorm_room','dining_room','music_studio','art_studio','parking_lot','street','tower','fastfood_restaurant','train_railway','physics_laboratory','arrival_gate/outdoor','gas_station']
# Object Names
# things dataset
things_plus_size_mean_matrix = pd.read_csv(things_plus_size_mean_path, sep='\t', engine='python', encoding='utf-8')
#%%
# Match object with vector
import re

def remove_numbers(string):
    return re.sub(r'\d+', '', string)

# Things
list(things_plus_size_mean_matrix['uniqueID'])

things2vec = {}
for w in list(things_plus_size_mean_matrix['uniqueID']):
    w_nonum = remove_numbers(w).replace('-','_')
    if w_nonum in word2vec.keys():
        things2vec[id2context[w]] = word2vec[w_nonum]

#%%
# Scenes
scenes2vec = {}
for s in sen_scene_to_keep:
    ss = s.replace('/','_').replace('_outdoor','').replace('_indoor','')
    print(ss)
    if ss in word2vec.keys():
        scenes2vec[s] = word2vec[ss]

#%% By hand
d = {
 'fishnet_stockings': word2vec['stockings'], 
 'iceskate': word2vec['ice_skate'], 
 'orange_rind': word2vec['rind'], 
 'swing_set': word2vec['swingset']
 }
for k, v in d.items():
    things2vec[k] = v

d = {
 'physics_laboratory': word2vec['laboratory'], 
 'fastfood_restaurant': word2vec['fast_food_restaurant'], 
 'train_railway': word2vec['railway'], 
 'bakery/shop': word2vec['bakery'], 
 'cubicle/office': word2vec['cubicle']
 }
for k, v in d.items():
    scenes2vec[k] = v

#%%
# Save
import pickle
with open('things2vec.pkl', 'wb') as f:
    pickle.dump(things2vec, f)
with open('scenes2vec.pkl', 'wb') as f:
    pickle.dump(scenes2vec, f)

# %%
