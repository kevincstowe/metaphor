extra_root = "/data/kevin/GitHub/metaphor/add_ons/"
vn_api_root = "/data/kevin/GitHub/verbnet/"
model_loc = "/data/kevin/Vectors/models/"
parser_loc = "/data/kevin/parser/"

TAG_NOUNS = {"NNP", "NNS", "NN", "NNPS"}
TAG_ADJS = {"JJ", "JJR", "JJS"}
TAG_ADVS = {"RB", "RBR", "RBS"}
TAG_VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
TAG_PREPS = ["IN"]
TAG_ALL = {'LS', 'VBZ', 'WP', 'WRB', 'RB', 'FW', 'RBS', 'PRP$', 'JJR', 'RBR', 'VBG', 'NNPS', 'DT', 'NNS', 'EX', 'IN', 'UH',
       'CC', 'PDT', 'NN', 'TO', 'VB', 'PRP', 'RP', 'JJS', 'JJ', 'VBD', 'MD', 'VBP', 'CD', 'POS', 'WDT', 'VBN', 'WP$',
       'SYM', 'NNP'}

VUAMC_NOUNS = {"NN0", "NN1", "NN2", "NP0", "PNI", "PNX"}
VUAMC_ADJS = {"AJ0", "AJC", "AJS"}
VUAMC_ADVS = {"AV0", "AVP", "AVQ"}
VUAMC_VERBS = {"VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ"}
VUAMC_PREPS = {"PRF", "PRP"}
VUAMC_ALL = ['PNP', 'VHD', 'PUL', 'CRD', 'NN1-VVG', 'NP0', 'None', 'CJS', 'CJC', 'VDB', 'PNI-CRD', 'DPS', 'AJ0-VVD', 'PRF', 'VVG', 'NN1-AJ0', 'NN2', 'AJC', 'AJ0', 'VDI', 'VDN', 'VHB', 'AJ0-VVN', 'VVD-AJ0', 'AVQ', 'CRD-PNI', 'PUN', 'VDG', 'CJT-DT0', 'VVN-VVD', 'VVD-VVN', 'NN1-VVB', 'UNC', 'VVN', 'DT0-CJT', 'VBG', 'VVZ', 'VHG', 'PUR', 'VBI', 'VVD', 'AJ0-VVG', 'TO0', 'VVI', 'CJT', 'XX0', 'VBB', 'POS', 'NP0-NN1', 'AV0', 'VHZ', 'VVG-AJ0', 'ITJ', 'VHI', 'PRP-AVP', 'PRP-CJS', 'PNQ', 'CJS-AVQ', 'NN2-VVZ', 'NN0', 'AJ0-NN1', 'VVB', 'AV0-AJ0', 'NN1-NP0', 'VHN', 'ORD', 'VVN-AJ0', 'AT0', 'VBZ', 'VDD', 'AJ0-AV0', 'DTQ', 'sentence', 'AJS', 'PNI', 'VVG-NN1', 'VVZ-NN2', 'PUQ', 'DT0', 'AVP-PRP', 'VVB-NN1', 'PRP',
'CJS-PRP', 'VBN', 'VBD', 'VDZ', 'NN1', 'AVP', 'EX0', 'AVQ-CJS', 'ZZ0', 'PNX', 'VM0']

ACADEMIC = {'b17-fragment02', 'clw-fragment01', 'ecv-fragment05', 'acj-fragment01', 'a6u-fragment02', 'crs-fragment01', 'b1g-fragment02', 'ew1-fragment01', 'fef-fragment03', 'clp-fragment01', 'amm-fragment02', 'alp-fragment01', 'as6-fragment01', 'as6-fragment02', 'ea7-fragment03'}
CONVERSATION = {'kbw-fragment09', 'kb7-fragment31', 'kb7-fragment48', 'kbd-fragment21', 'kb7-fragment45','kbh-fragment04', 'kcv-fragment42', 'kbp-fragment09', 'kcf-fragment14', 'kcu-fragment02', 'kbw-fragment42', 'kbd-fragment07', 'kbh-fragment01', 'kb7-fragment10', 'kcc-fragment02', 'kbh-fragment02', 'kbh-fragment09', 'kbh-fragment03', 'kbw-fragment04', 'kbw-fragment17', 'kbj-fragment17', 'kbw-fragment11', 'kbh-fragment41', 'kbc-fragment13'}
FICTION = {'faj-fragment17', 'ccw-fragment04', 'cty-fragment03', 'bmw-fragment09', 'fpb-fragment01', 'c8t-fragment01', 'cdb-fragment04', 'fet-fragment01', 'ab9-fragment03', 'ac2-fragment06', 'bpa-fragment14', 'g0l-fragment01', 'cb5-fragment02', 'ccw-fragment03', 'cdb-fragment02'}
NEWS = {'al2-fragment16', 'ahd-fragment06', 'a3e-fragment02', 'ahc-fragment61', 'a36-fragment07', 'a3m-fragment02', 'a5e-fragment06', 'a1j-fragment33', 'ahe-fragment03', 'aa3-fragment08', 'a7w-fragment01', 'a31-fragment03', 'a1u-fragment04', 'a7t-fragment01', 'a80-fragment15', 'a1n-fragment09', 'a1e-fragment01', 'a3p-fragment09', 'a1m-fragment01', 'a1l-fragment01', 'a8m-fragment02', 'a8n-fragment19', 'a98-fragment03', 'a1g-fragment27', 'ahb-fragment51', 'ahf-fragment24', 'a1f-fragment09', 'a3e-fragment03', 'a1f-fragment12', 'ahc-fragment60', 'a1h-fragment05', 'a8r-fragment02', 'a1f-fragment08', 'a3c-fragment05', 'a1p-fragment03', 'al5-fragment03', 'a9j-fragment01', 'al0-fragment06', 'a2d-fragment05', 'a7y-fragment03', 'a1n-fragment18', 'a1f-fragment10', 'a1f-fragment06', 'a1x-fragment05', 'a4d-fragment02', 'a7s-fragment03', 'ajf-fragment07', 'a1h-fragment06', 'al2-fragment23', 'a1p-fragment01', 'a1g-fragment26', 'a1f-fragment11', 'a3k-fragment11', 'a1k-fragment02', 'a39-fragment01', 'ahf-fragment63', 'a1j-fragment34', 'a1x-fragment04', 'a1x-fragment03', 'a38-fragment01', 'ahl-fragment02', 'a8u-fragment14', 'a1f-fragment07'}

DOMAINS = ['MARRIAGE', 'VEHICLE', 'DEMOGRAPHICS', 'THEFT', 'HUMAN_BODY', 'LIGHT', 'BACKWARD_MOVEMENT', 'BARRIER', 'DESIRE', 'UPWARD_MOVEMENT', 'RESOURCE', 'STORY', 'VISION', 'BATTLE', 'A_GOD', 'ADDICTION', 'DESTROYER', 'ELECTIONS', 'MAZE', 'TEMPERATURE', 'GUN_OWNERSHIP', 'CONTROL_OF_GUNS', 'PATHWAY', 'MIGRATION', 'SHAPE', 'HAZARDOUS_GEOGRAPHIC_FEATURE', 'POSITION_AND_CHANGE_OF_POSITION_ON_A_SCALE', 'DRUG_TRAFFICKING', 'BLOOD_STREAM', 'TERRORISM', 'GAP', 'ABYSS', 'PARASITE', 'MOVEMENT', 'TAXPAYERS', 'ACCIDENT', 'GREED', 'PHYSICAL_LOCATION', 'ABORTION', 'POLITICIANS', 'WEALTH', 'PLANT', 'BUILDING', 'FORWARD_MOVEMENT', 'LOW_POINT', 'FAMILY', 'HIGH_POINT', 'OBESITY', 'SCHISM', 'OBJECT_HANDLING', 'MEDICINE', 'SIZE', 'ENSLAVEMENT', 'PORTAL', 'CONTAINER', 'DEMOCRACY', 'JOURNEY', 'NATURAL_PHYSICAL_FORCE', 'SCIENCE', 'WAR', 'FOOD', 'STRUGGLE', 'GAME', 'WEAKNESS', 'GUN_DEBATE_GROUPS', 'GUN_VIOLENCE', 'FACTORY', 'TAXATION', 'FIRE', 'A_RIGHT', 'ISLAMIC', 'MACHINE', 'RELIGION', 'FORCEFUL_EXTRACTION', 'INSANITY', 'GIFT', 'ENERGY', 'CLIMATE_CHANGE', 'SERVANT', 'INTELLECTUAL_PROPERTY', 'RULE_ENFORCER', 'PROTECTION', 'EMOTION_EXPERIENCER', 'POVERTY', 'STAGE', 'PHYSICAL_BURDEN', 'MORAL_DUTY', 'CROP', 'MENTAL_CONCEPTS', 'TAXES', 'TOOL', 'DEBT', 'BODY_OF_WATER', 'COMPETITION', 'FABRIC', 'PHYSICAL_OBJECT', 'AVERSION', 'GEOGRAPHIC_FEATURE', 'STRENGTH', 'CONFINEMENT', 'DARKNESS', 'OTHER', 'BUREAUCRACY', 'WEATHER', 'FURNISHINGS', 'LOW_LOCATION', 'GOAL_DIRECTED', 'LEADER', 'INDUSTRY', 'LIFE_STAGE', 'MOVEMENT_ON_A_VERTICAL_SCALE', 'IMPURITY', 'HIGH_LOCATION', 'GUNS', 'CRIME', 'BUSINESS', 'WELFARE', 'DOWNWARD_MOVEMENT', 'ANIMAL', 'GOVERNMENT', 'CLOTHING', 'MAGIC', 'EMPLOYEE', 'DISEASE', 'BLOOD_SYSTEM', 'MONEY', 'PHYSICAL_HARM', 'MONSTER', 'PLIABILITY', 'GUN_RIGHTS', 'CONTAMINATION']

