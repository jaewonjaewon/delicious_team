#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:15:39 2021

@author: jiyeonseo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
from datetime import *
from sklearn.ensemble import RandomForestClassifier
import time


# 추가로 알고싶은 계산치는 함수 짜서 num_cal에 넣으면 됨.
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
def mode(x):
    return x.value_counts().index[0]

cat_cal = ['count','nunique',mode]
num_cal = ['sum','mean','median','min','max',mode,'std',q25,q75]

# 코드 도는지 확인 용도로 간단 계산만 해봄.
cat_cal = 'count'
num_cal=['sum','mean']


def info_append(uno_info,df,group_index,cal_dict):
    
    """
    Args:
        uno_info: left_join 당할 dataframe
        df: groupby로 계산할 dataframe
        group_index: groupby 기준이 될 column name or list
        cal_dict: df를 groupby한 뒤 각 column을 어떻게 계산할건지(예:평균, 합) 나타내는 dict
    Returns:
        join한 dataframe
    """
    
    grouped = df.groupby(group_index)
    dat = grouped.agg(cal_dict)
    dat.columns = dat.columns.to_flat_index()
    return pd.merge(left=uno_info,right=dat,on='uno',how='left')


class DataParser():
    def __init__(self,f_service,f_bookmark,f_coin,f_movie):
        
        df_service = pd.read_csv(f_service, parse_dates=['registerdate','enddate'], infer_datetime_format=True)
        df_service['gender'] = df_service['gender'].fillna('N')
        df_service['agegroup'] = df_service['agegroup'].replace(950, 0)
        df_service.loc[(df_service['pgamount'] <  100), 'pgamount'] = df_service['pgamount'] * 1120
        df_service = df_service.fillna('X')
        df_service.Repurchase = np.where(df_service.Repurchase == 'X', 1, 0)
        self.service = df_service
        
        movie_info = pd.read_csv(f_movie)
        movie_info['running_time'] = movie_info.apply(lambda row: float(str(row['runningtime']).rstrip('분'))*60,axis=1)
        movie_info.loc[9943,['runningtime','running_time']]=[136,136*60]
        self.movie = movie_info
        
        df_bookmark = pd.read_csv(f_bookmark, parse_dates=['dates'], infer_datetime_format=True)
        df_bookmark['dates_time'] = df_bookmark.apply(lambda row: pd.Timestamp(datetime.combine(datetime.date(row['dates']),datetime.strptime(str(row['hour']),'%H').time())),axis=1)
        df_bookmark = pd.merge(left=df_bookmark,right=movie_info[['movieid','running_time']],how='left',left_on='contentid',right_on='movieid')
        df_bookmark['dates_sec'] = df_bookmark.dates_time.astype(int)/10**9
        self.bookmark = df_bookmark
        
        self.coin = pd.read_csv(f_coin, parse_dates=['registerdate'], infer_datetime_format=True)
        
        self.uno_info = pd.DataFrame(np.unique(df_service.uno),columns = ['uno'])
        

    def bookmark_parsing(self):
        df_bookmark = self.bookmark
        uno_info = self.uno_info
        
        cal_dict = {'channeltype':cat_cal,
            'devicetype':cat_cal,
            'hour':num_cal,
            'dates_sec':num_cal,
            'viewtime':num_cal
            }
        uno_info = info_append(uno_info,df_bookmark,'uno',cal_dict)
        
        cal_dict = {'programid':cat_cal,
                    'contentid':cat_cal,
                    'hour':num_cal,
                    'viewtime':num_cal}
        uno_info = info_append(uno_info,df_bookmark[df_bookmark.channeltype=='V'],'uno',cal_dict)
        
        
        cal_dict = {'programid':cat_cal,
                    'hour':num_cal,
                   'viewtime':num_cal}
        uno_info = info_append(uno_info,df_bookmark[df_bookmark.channeltype=='E'],'uno',cal_dict)
        
        
        cal_dict = {'programid':cat_cal,
                    'hour':num_cal,
                   'viewtime':num_cal}
        uno_info = info_append(uno_info,df_bookmark[df_bookmark.channeltype=='L'],'uno',cal_dict)
        
        
        df_movie = df_bookmark[df_bookmark.channeltype=='M']
        df_movie['total_viewtime'] = df_bookmark[df_bookmark.channeltype=='M'][['uno','contentid','viewtime']].groupby(by=['uno','contentid']).transform(np.sum)
        df_movie['watch_ratio'] = df_movie.apply(lambda row: row['total_viewtime']/row['running_time'],axis=1)
        cal_dict = {'total_viewtime':num_cal,
                    'watch_ratio':num_cal}
        uno_info = info_append(uno_info,df_movie,'uno',cal_dict)
        
        
        cal_dict = {'contentid':cat_cal,
                    'viewtime':num_cal}
        uno_info = info_append(uno_info,df_bookmark[df_bookmark.channeltype=='M'],'uno',cal_dict)
                
        self.uno_info = uno_info
                
        
    def service_parsing(self):
        df_service = self.service
        uno_info = self.uno_info
        
        cal_dict = {'pgamount':num_cal,
        'devicetypeid':cat_cal,
        'productcode':cat_cal}
        
        uno_info = info_append(uno_info,df_service,'uno',cal_dict)

        self.uno_info = uno_info
        
    
    def coin_parsing(self):
        df_coin = self.coin
        uno_info = self.uno_info
        
        
        cal_dict = {'productcode':cat_cal,
            'totalamount':num_cal,
            'pgamount':num_cal,
            'coinamount':num_cal,
            'bonusamount':num_cal,
            'discountamount':num_cal}

        uno_info = info_append(uno_info,df_coin,'uno',cal_dict)
        
        self.uno_info = uno_info
        


if __name__ == "__main__":
    parser = DataParser('train_service.csv','train_bookmark.csv','coin.csv','movie_info.csv')
    parser.bookmark_parsing()
    parser.service_parsing()
    parser.coin_parsing()
    
    parser.uno_info.to_csv('train_uno_info.csv')
    
    
    parser = DataParser('predict_service.csv','predict_bookmark.csv','coin.csv','movie_info.csv')
    parser.bookmark_parsing()
    parser.service_parsing()
    parser.coin_parsing()
    dd = parser.uno_info
    
    parser.uno_info.to_csv('predict_uno_info.csv')
