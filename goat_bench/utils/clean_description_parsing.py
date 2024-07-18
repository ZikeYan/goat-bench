from json import load
from pickle import NONE
from goat_bench.utils.utils import write_txt, load_json, write_json
all_cats = []
# invalid_cats = [
#     'america', 'animals', 'appearance', 'attention', 'boundaries', 'bounding box', 
#     'boy bedroom', 'brick building', 'building', 'child room', 'children book', 
#     'church', 'coordinates', 'deer', 'deer head', 'direction', 'directions', 
#     'environment', 'explanation', 'figure', 'group', 'gulf', 'image', 
#     'information', 'items', 'lake', 'landscape', 'level', 'location', 'lot', 
#     'lots', 'man', 'mascot', 'maurin quina', 'men', 'mountain bike', 
#     'new york city skyline', 'parapet', 'people', 'philodendron sempervirens', 
#     'position', 'positions', 'priest', 'proximity', 'region', 'region coordinates', 
#     'region semantics', 'relative', 'repica', 'rest', 'robot agent', 'scene', 
#     'size', 'space', 'stories', 'stuff', 'target object', 'terms', 'thorns', 
#     'typewriter', 'united states', 'view', 'way', 'woman', 'women', 
#     'world map', 'zombie'
# ]
# NOte: For 'door', 'wall', 'glass door', 'floor' etc., should not be supervised
invalid_cats = [
    'america', 'animals', 'appearance', 'attention', 'boundaries', 'bounding box', 
    'boy bedroom', 'brick building', 'building', 'child room',  
    'church', 'coordinates', 'deer', 'deer head', 'direction', 'directions', 
    'environment', 'explanation', 'figure', 'group', 'gulf', 'image', 
    'information', 'items', 'lake', 'landscape', 'level', 'location', 'lot', 
    'lots', 'man', 'mascot', 'maurin quina', 'men', 
    'new york city skyline', 'number', 'objects', 'people', 'philodendron sempervirens', 
    'position', 'positions', 'priest', 'proximity', 'region', 'region coordinates', 
    'region semantics', 'relative', 'repica', 'rest', 'robot agent', 'scene', 'set',
    'size', 'space', 'stories', 'stuff', 'target object', 'terms', 'thorns', 
    'typewriter', 'united states', 'view', 'way', 'woman', 
    'world map', 'zombie'
]
def sort_substrings_by_sequence(substr_list, main_str):
    # Sort substr_list based on the index of each substring in main_str
    sorted_substr_list = sorted(substr_list, key=lambda substr: main_str.find(substr))
    return sorted_substr_list

# for task_set in ['val_seen', 'val_unseen', 'val_seen_synonyms']:
#     lang_pase_path = f'data/languagenav/lang_parse/parse_{task_set}_spacy.json'
#     lang_pase_cleaned_path = f'data/languagenav/lang_parse/parse_{task_set}_spacy_cleaned.json'
#     info  = load_json(lang_pase_path)
    # cleaned_dict = {}
    # print("*"*40, task_set, "*"*40)
    # for k, v in info.items():
    #     cleaned_dict[k] = {}
    #     cleaned_dict[k]["adjs"] = v["adjs"]
    #     cleaned_dict[k]['nearby'] = [cat for cat in v['nearby'] if cat not in invalid_cats]
    #     if v['target'] in invalid_cats:
    #         print(k, v['target'], v['nearby'])
    #         sorted_nearby = sort_substrings_by_sequence(v['nearby'], k)
    #         cleaned_dict[k]['target'] = sorted_nearby.pop(0)
    #         cleaned_dict[k]['nearby'] = sorted_nearby
    #         print(cleaned_dict[k]['target'], cleaned_dict[k]['nearby'])
    #     else:
    #         cleaned_dict[k]['target'] = v['target']
        
    # write_json(cleaned_dict, lang_pase_cleaned_path)
    # for value_dict in info.values():
    #     all_cats.append(value_dict['target'])
    #     all_cats.extend(value_dict['nearby'])
    # image_pase_path = f'data/imagenav/image_goal_caption_parse/{task_set}_caption_parse_spacy_cleaned.json'
#     info  = load_json(lang_pase_cleaned_path)
#     for value_dict in info.values():
#         all_cats.append(value_dict['target'])
#         all_cats.extend(value_dict['nearby'])
# write_txt(sorted(list(set(all_cats))), 'data/languagenav/lang_parse/lang_all_cats.txt')
# invalid_cats = ["bear", "bird", "branches", "building", "city", "cow", "cow head", "deer head", "dog", "dog head", "feather", "horse", "husband", "jesus", "man", "middle", "monitors", "mountains", "nativity", "people", "plane", "room", "space station", "starfish", "swan", "top", "train", "train station", "trees", "view", "virgin mary", "water", "woman"]
tmp = load_json('/home/yan/Workspace/giftednav_ws/src/GiftedNav/scripts/dataloader/object_info/object_to_synonym.json').keys()
for task_set in ['val_seen', 'val_unseen', 'val_seen_synonyms']:
    lang_pase_cleaned_path = f'data/languagenav/lang_parse/parse_{task_set}_spacy_cleaned.json'
    info  = load_json(lang_pase_cleaned_path)
    for value_dict in info.values():
        if value_dict['target'] is not None:
            all_cats.append(value_dict['target'])
            all_cats.extend(value_dict['nearby'])
                
    lang_pase_path = f'data/imagenav/image_goal_caption_parse/{task_set}_caption_parse_spacy_blip2.json'
    lang_pase_cleaned_path = f'data/imagenav/image_goal_caption_parse/{task_set}_caption_parse_spacy_blip2_cleaned.json'
    info  = load_json(lang_pase_cleaned_path)
    
#     cleaned_dict = {}
#     print("*"*40, task_set, "*"*40)
#     for k, v in info.items():
#         cleaned_dict[k] = {}
#         cleaned_dict[k]["adjs"] = v["adjs"]
#         cleaned_dict[k]['nearby'] = [cat for cat in v['nearby'] if cat not in invalid_cats]
#         if v['target'] in invalid_cats:
#             print(k, v['target'], v['nearby'])
#             if len(v['nearby'])!=0:
#                 sorted_nearby = sort_substrings_by_sequence(v['nearby'], k)
#                 cleaned_dict[k]['target'] = sorted_nearby.pop(0)
#                 cleaned_dict[k]['nearby'] = sorted_nearby
#             else:
#                 cleaned_dict[k]['target'] = None
#                 cleaned_dict[k]['nearby'] = []
#             print(cleaned_dict[k]['target'], cleaned_dict[k]['nearby'])
#         else:
#             cleaned_dict[k]['target'] = v['target']
        
#     write_json(cleaned_dict, lang_pase_cleaned_path)
    for value_dict in info.values():
        if value_dict['target'] is not None:
            all_cats.append(value_dict['target'])
            all_cats.extend(value_dict['nearby'])
write_txt(sorted(list(set(all_cats))), 'data/lang_image_parse_all_cats.txt')