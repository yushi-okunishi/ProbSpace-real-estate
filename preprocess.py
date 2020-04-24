from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import MeCab
from tqdm import tqdm

def rename_columns(df):
    col_name_map = {
        '種類': "type", 
        '地域': "region", 
        '市区町村コード': "city_code",
        '都道府県名': "prefecture_name",
        '市区町村名': "city_name", 
        '地区名': "district_name", 
        '最寄駅：名称': "nearest_station_name",
        '最寄駅：距離（分）': "nearest_station_distance",
        '間取り': "floor_plan", 
        '面積（㎡）': "area", 
        '土地の形状': "land_shape",
        '間口': "frontage", 
        '延床面積（㎡）': "floor_area", 
        '建築年': "year_of_construction", 
        '建物の構造': "building_structure", 
        '用途': "use", 
        '今後の利用目的': "future_purpose", 
        '前面道路：方位': "front_road_direction", 
        '前面道路：種類': "front_road_type", 
        '前面道路：幅員（ｍ）': "front_road_width",
        '都市計画': "city_planning", 
        '建ぺい率（％）': "building_coverage_ratio", 
        '容積率（％）': "floor_area_ratio", 
        '取引時点': "transaction_point", 
        '改装': "renovation", 
        '取引の事情等': "transaction_circumstances"
    }
    return df.rename(columns=col_name_map)

# 各項目についての前処理

def p_area(v):
    if v in ['2000㎡以上', '5000㎡以上']:
        area = int(v[:4])
    else:
        area = int(v)
    return area

def is_area_over(v):
    if v in ['2000㎡以上', '5000㎡以上']:
        return 1
    else:
        return 0

def p_frontage(v):
    if v == 'NULL':
        return None
    if v == '50.0m以上':
        return 50.0
    return float(v)

def is_frontage_over(v):
    if v == '50.0m以上':
        return 1
    else:
        return 0

def p_floor_area(v):
    if v == 'NULL':
        return None
    elif v == '2000㎡以上':
        return 2000
    elif v == '10m^2未満':
        return 0
    else:
        int(v)

def is_floor_area_over(v):
    if v == '2000㎡以上':
        return 1
    elif v == '10m^2未満':
        return -1
    else:
        return 0
    
def nb_R(v):
    other = ['オープンフロア', 'NULL', 'スタジオ', 'メゾネット']
    if v in other:
        return 0
    else:
        return int(v[0])
    
def is_other_fp(v):
    if v == 'NULL':
        return None
    elif v == 'スタジオ':
        return 1
    elif v == 'オープンフロア':
        return 2
    elif v == 'メゾネット':
        return 3
    else:
        return 0
    
def has_D(v):
    return 1 if 'Ｄ' in v else 0
    
def has_L(v):
    return 1 if 'Ｌ' in v else 0
    
def has_K(v):
    return 1 if 'Ｋ' in v else 0
    
def has_pK(v):
    return 1 if '＋Ｋ' in v else 0
    
def has_S(v):
    return 1 if 'Ｓ' in v else 0

def p_tp_year(v):
    return int(v[:4])

def p_tp_month(v):
    return int(v[6])

def p_nearest_station(v):
    if v == 'NULL':
        return None
    elif v == '30分?60分':
        return 45
    elif v == '1H30?2H':
        return 105
    elif v == '1H?1H30':
        return 75
    elif v == '2H?':
        return 120
    else:
        return int(v)
    
def p_year_of_const(v):
    if v == 'NULL':
        return None
    elif v == '戦前':
        return 5
    elif v[:2] == '昭和':
        return int(v[2:-1])
    else:
        return int(v[2:-1]) + 63
    
def is_use_for_parking(v):
    if '駐車場' in v:
        return 1
    else:
        return 0
    
def is_use_for_workplace(v):
    if '作業場' in v:
        return 1
    else:
        return 0
    
def is_use_for_office(v):
    if '事務所' in v:
        return 1
    else:
        return 0
    
def is_use_for_other(v):
    if 'その他' in v:
        return 1
    else:
        return 0
    
def is_use_for_warehouse(v):
    if '倉庫' in v:
        return 1
    else:
        return 0
    
def is_use_for_residential(v):
    if '住宅' in v:
        return 1
    else:
        return 0
    
def is_use_for_factory(v):
    if '工場' in v:
        return 1
    else:
        return 0
def is_use_for_apartment(v):
    if '共同住宅' in v:
        return 1
    else:
        return 0
    
def is_use_for_store(v):
    if '店舗' in v:
        return 1
    else:
        return 0

# SRC RC "Steel frame" "Wooden" "Steel frame, lightweight steel frame"
def is_bs_ＳＲＣ(v):
    return 1 if "ＳＲＣ" in v else 0

def is_bs_RC(v):
    for i in range(len(v)-1):
        if v[i] == "Ｒ" and v[i+1] == 'Ｃ' and (i == 0 or v[i-1] != 'Ｓ'):
            return 1
    return 0

def is_bs_steel(v):
    for i in range(len(v)-1):
        if v[i] == "鉄" and v[i+1] == '骨' and (i == 0 or v[i-1] != '量'):
            return 1
    return 0

def is_bs_lightweight_steel(v):
    return 1 if "軽量鉄骨" in v else 0

def is_bs_wooden(v):
    return 1 if "木造" in v else 0

def is_bs_block(v):
    return 1 if "ブロック" in v else 0

# transaction circumstances
# '私道を含む取引' '隣地の購入' '関係者間取引' '調停・競売等', 'その他事情有り', '他の権利・負担付き', '瑕疵有りの可能性', '古屋付き・取壊し前提'

def is_tcc_private(v):
    return 1 if '私道を含む取引' in v else 0

def is_tcc_next(v):
    return 1 if '隣地の購入' in v else 0

def is_tcc_related(v):
    return 1 if '関係者間取引' in v else 0

def is_tcc_other(v):
    return 1 if 'その他事情有り' in v else 0

def is_tcc_auction(v):
    return 1 if '調停・競売等' in v else 0

def is_tcc_other_rights(v):
    return 1 if '他の権利' in v else 0

def is_tcc_defects(v):
    return 1 if '瑕疵有り' in v else 0

def is_tcc_old(v):
    return 1 if '古屋付き' in v else 0
    
# 最終的な前処理
def mp(train_df, test_df, kf, log1=False):
    dfs = [train_df, test_df]
    del train_df
    del test_df
    
    for i in range(len(dfs)):
        dfs[i] = rename_columns(dfs[i])
        
    x_col = []
    y_col = []
    for c in dfs[0].columns:
        if c ==  'y':
            y_col.append(c)
        elif c != 'id':
            x_col.append(c)
    x_col.remove('city_name') # city_codeと同じ
    x_col.remove('prefecture_name') # 1 unique
    #print(x_col, y_col)
    p_col = ['area', 'frontage', 'floor_area', 'transaction_point']
    init_col = ['area_over', 'frontage_over', 'floor_area_over',
                'transaction_point_year', 'transaction_point_month',
                'nb_R', 'has_L', 'has_D', 'has_K', 'has_pK', 'has_S', 'other_fp', #floor plan
                'is_use_parking', 'is_use_workplace', 'is_use_office', 'is_use_other', 'is_use_warehouse',
                'is_use_residential', 'is_use_factory', 'is_use_apartment', 'is_use_store', #use for
                'is_bs_RC', 'is_bs_SRC', 'is_bs_steel', 'is_bs_wooden', 'is_bs_lw_steel', 'is_bs_block', #building structure
                'is_tcc_private', 'is_tcc_next', 'is_tcc_related', 'is_tcc_other', 'is_tcc_auction',
                'is_tcc_other_rights', 'is_tcc_defects', 'is_tcc_old', # transaction circumstances
                'log1_area',
               ]
    # Parking lot', 'Workplace', 'Office', 'Other', 'Warehouse', 'Residential', 'Factory', 'Apartment', 'Store'
    for c in x_col:
        if dfs[0][c].dtype == 'object':
            for i in range(2):
                dfs[i][c] = dfs[i][c].fillna("NULL")
                
    #p1_col = p_col[:3]
    scaler_dict = {}
    for i in range(len(dfs)):
        if i == 0:
            print('train')
        else:
            print('test')
            
        print('processing area')
        dfs[i]['area_over'] = dfs[i]['area'].map(is_area_over)
        dfs[i]['area'] = dfs[i]['area'].map(p_area)
        dfs[i]['log1_area'] = dfs[i]['area'].map(np.log1p)
        print('processing frontage')
        dfs[i]['frontage_over'] = dfs[i]['frontage'].map(is_frontage_over)
        dfs[i]['frontage'] = dfs[i]['frontage'].map(p_frontage)
        
        print("scale and absolute")
        star = ['log1_area', 'frontage', 'front_road_width']
        std_star = ['std_{}'.format(c) for c in star]
        if i == 0:
            scaler = StandardScaler()
            for c in std_star:
                dfs[i][c] = 0
            dfs[i][std_star] = scaler.fit_transform(dfs[i][star])
        else:
            for c in std_star:
                dfs[i][c] = 0
            dfs[i][std_star] = scaler.transform(dfs[i][star])
        abs_col = []
        for k in range(len(std_star)):
            for l in range(k+1, len(std_star)):
                col_name = 'abs_{}_{}'.format(std_star[k], std_star[l])
                abs_col.append(col_name)
                dfs[i][col_name] = dfs[i][std_star[k]].values - dfs[i][std_star[l]].values
                
        print('processing floor area')
        dfs[i]['floor_area_over'] = dfs[i]['floor_area'].map(is_floor_area_over)
        dfs[i]['floor_area'] = dfs[i]['floor_area'].map(p_floor_area)
        print('processing transaction point')
        dfs[i]['transaction_point_year'] = dfs[i]['transaction_point'].map(p_tp_year)
        dfs[i]['transaction_point_month'] = dfs[i]['transaction_point'].map(p_tp_month)
        print('processing floor_plan')
        dfs[i]['nb_R'] = dfs[i]['floor_plan'].map(nb_R)
        dfs[i]['has_L'] = dfs[i]['floor_plan'].map(has_L)
        dfs[i]['has_D'] = dfs[i]['floor_plan'].map(has_D)
        dfs[i]['has_K'] = dfs[i]['floor_plan'].map(has_K)
        dfs[i]['has_pK'] = dfs[i]['floor_plan'].map(has_pK)
        dfs[i]['has_S'] = dfs[i]['floor_plan'].map(has_S)
        dfs[i]['other_fp'] = dfs[i]['floor_plan'].map(is_other_fp)
        print('processing nearest station distance')
        dfs[i]['nearest_station_distance'] = dfs[i]['nearest_station_distance'].map(p_nearest_station)
        print('processing year of construction')
        dfs[i]['year_of_construction'] = dfs[i]['year_of_construction'].map(p_year_of_const)
        print('processing use')
        dfs[i]['is_use_parking'] = dfs[i]['use'].map(is_use_for_parking)
        dfs[i]['is_use_workplace'] = dfs[i]['use'].map(is_use_for_workplace)
        dfs[i]['is_use_office'] = dfs[i]['use'].map(is_use_for_office)
        dfs[i]['is_use_other'] = dfs[i]['use'].map(is_use_for_other)
        dfs[i]['is_use_warehouse'] = dfs[i]['use'].map(is_use_for_warehouse)
        dfs[i]['is_use_residential'] = dfs[i]['use'].map(is_use_for_residential)
        dfs[i]['is_use_factory'] = dfs[i]['use'].map(is_use_for_factory)
        dfs[i]['is_use_apartment'] = dfs[i]['use'].map(is_use_for_apartment)
        dfs[i]['is_use_store'] = dfs[i]['use'].map(is_use_for_store)
        print("processing building structure")
        dfs[i]['is_bs_RC'] = dfs[i]['building_structure'].map(is_bs_RC)
        dfs[i]['is_bs_SRC'] = dfs[i]['building_structure'].map(is_bs_SRC)
        dfs[i]['is_bs_steel'] = dfs[i]['building_structure'].map(is_bs_steel)
        dfs[i]['is_bs_lw_steel'] = dfs[i]['building_structure'].map(is_bs_lightweight_steel)
        dfs[i]['is_bs_wooden'] = dfs[i]['building_structure'].map(is_bs_wooden)
        dfs[i]['is_bs_block'] = dfs[i]['building_structure'].map(is_bs_block)
        print('preprocessing transaction circumstances')
        """
        'is_tcc_private', 'is_tcc_next', 'is_tcc_related', 'is_tcc_other', 'is_tcc_auction',
        'is_tcc_other_rights', 'is_tcc_defects', 'is_tcc_old'
        """
        dfs[i]['is_tcc_private'] = dfs[i]['transaction_circumstances'].map(is_tcc_private)
        dfs[i]['is_tcc_next'] = dfs[i]['transaction_circumstances'].map(is_tcc_next)
        dfs[i]['is_tcc_related'] = dfs[i]['transaction_circumstances'].map(is_tcc_related)
        dfs[i]['is_tcc_other'] = dfs[i]['transaction_circumstances'].map(is_tcc_other)
        dfs[i]['is_tcc_auction'] = dfs[i]['transaction_circumstances'].map(is_tcc_auction)
        dfs[i]['is_tcc_other_rights'] = dfs[i]['transaction_circumstances'].map(is_tcc_other_rights)
        dfs[i]['is_tcc_defects'] = dfs[i]['transaction_circumstances'].map(is_tcc_defects)
        dfs[i]['is_tcc_old'] = dfs[i]['transaction_circumstances'].map(is_tcc_old)
    x_col.remove('transaction_point')
    x_col.remove('floor_plan')
    x_col.remove('use')
    x_col.remove('building_structure')
    x_col.remove('transaction_circumstances')
    #x_col += std_star
    x_col += abs_col
    
    # target encoding
    
    target_encoding_col = [
        'type', 'region', 'city_code', 'district_name', 'nearest_station_name', 'land_shape', 'year_of_construction',
        'future_purpose', 'city_planning', 'nb_R', 'has_L', 'has_D', 'has_K', 'has_pK', 'has_S'
    ]
    
    
    # targetをlog1p変換
    if log1:
        print("target log1p()")
        for i in range(2):
            dfs[i]['log1_y'] = np.log1p(dfs[i]['y'].values)
        
    
    print("target, min, max. encoding")
    for c in target_encoding_col:
        tm_col = '{}_target_mean'.format(c)
        tv_col = '{}_target_var'.format(c)
        tmax = '{}_target_max'.format(c)
        tmin = '{}_target_min'.format(c)
        x_col += [tm_col, tv_col, tmax, tmin]
        for i in range(2):
            dfs[i][tm_col] = 0
            dfs[i][tv_col] = 0
            dfs[i][tmax] = 0
            dfs[i][tmin] = 0
    for train_ind, val_ind in kf.split(dfs[0][x_col], dfs[0][y_col]):
        for c in target_encoding_col:
            tm_col = '{}_target_mean'.format(c)
            tv_col = '{}_target_var'.format(c)
            tmax = '{}_target_max'.format(c)
            tmin = '{}_target_min'.format(c)
            grouped = dfs[0].iloc[train_ind].groupby(c)
            if log1:
                target_mean = grouped.log1_y.mean()
                target_var = grouped.log1_y.var()
                target_max = grouped.log1_y.max()
                target_min = grouped.log1_y.min()
            else:
                target_mean = grouped.y.mean()
                target_var = grouped.y.var()
                target_max = grouped.y.max()
                target_min = grouped.y.min()
            #df['target_enc'] = df['city'].map(target_mean)
            dfs[0][tm_col].iloc[val_ind] = dfs[0][c].iloc[val_ind].map(target_mean)
            dfs[0][tv_col].iloc[val_ind] = dfs[0][c].iloc[val_ind].map(target_var)
            dfs[0][tmax].iloc[val_ind] = dfs[0][c].iloc[val_ind].map(target_max)
            dfs[0][tmin].iloc[val_ind] = dfs[0][c].iloc[val_ind].map(target_min)
    for c in target_encoding_col:
        tm_col = '{}_target_mean'.format(c)
        tv_col = '{}_target_var'.format(c)
        tmax = '{}_target_max'.format(c)
        tmin = '{}_target_min'.format(c)
        grouped = dfs[0].groupby(c)
        if log1:
            target_mean = grouped.log1_y.mean()
            target_var = grouped.log1_y.mean()
            target_max = grouped.log1_y.max()
            target_min = grouped.log1_y.min()
        else:
            target_mean = grouped.y.mean()
            target_var = grouped.y.mean()
            target_max = grouped.y.max()
            target_min = grouped.y.min()
        dfs[1][tm_col] = dfs[1][c].map(target_mean)
        dfs[1][tv_col] = dfs[1][c].map(target_var)
        dfs[1][tmax] = dfs[1][c].map(target_max)
        dfs[1][tmin] = dfs[1][c].map(target_min)
    
        
    print("area, frontage, front_road_width linear regression encoding")
    tar_col = ['std_log1_area', 'std_frontage', 'std_front_road_width']
    dfs[0]['linear_target'] = 0
    dfs[0]['linear_target'].loc[dfs[0][tar_col].dropna().index] = None
    dfs[1]['linear_target'] = 0
    dfs[1]['linear_target'].loc[dfs[1][tar_col].dropna().index] = None
    notna_df_test = dfs[1][tar_col + y_col].dropna()
    notna_df_test['linear_target'] = 0
    for train_ind, val_ind in kf.split(dfs[0][tar_col], dfs[0][y_col]):
        lr = LinearRegression()
        #np.zeros((len()))
        notna_df = dfs[0].iloc[train_ind][tar_col + y_col].dropna()
        lr.fit(notna_df[tar_col], notna_df[y_col])
        notna_df_val = dfs[0].iloc[val_ind][tar_col + y_col].dropna()
        notna_df_val['linear_target'] = lr.predict(notna_df_val[tar_col])
        dfs[0]['linear_target'].loc[notna_df_val.index] = notna_df_val['linear_target']
        test_pred = lr.predict(notna_df_test[tar_col])
        #print(test_pred.shape)
        notna_df_test['linear_target'] += test_pred[:, 0]
    dfs[1]['linear_target'].loc[notna_df_test.index] = notna_df_test['linear_target'] / 5
    x_col.append('linear_target')
        
    for c in x_col:
        if c == 'linear_target': continue
        if c in p_col:
            continue
        if dfs[0][c].dtype == 'object':
            le = LabelEncoder()
            #train_df[c].fillna("NULL")
            #dfs[[c] = train_df[c].fillna("NULL")
            #test_df[c] = test_df[c].fillna("NULL")
            le.fit(np.concatenate([dfs[0][c].values, dfs[1][c].values], axis=0))
            dfs[0][c] = le.transform(dfs[0][c])
            dfs[1][c] = le.transform(dfs[1][c])
            
    x_col += init_col
    light_gbm_col = x_col
    nn_col = x_col
            
    return dfs, x_col, y_col