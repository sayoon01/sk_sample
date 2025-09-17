# -*- coding:utf-8 -*-

import os
import pandas as pd
import sys
from collections import defaultdict
import fileinput
import numpy as np
import time
import datetime
import ray
import pickle

def printProgressBar(iteration, total, prefix = 'Progress', suffix = 'Complete',\
                      decimals = 1, length = 50, fill = '█'): 
    # 작업의 진행상황을 표시
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()
    if iteration == total:
        print()

def recursive_search_dir(_nowDir, _filelist):
    dir_list = []  # 현재 디렉토리의 서브디렉토리가 담길 list
    try:
        f_list = os.listdir(_nowDir)
    except FileNotFoundError:
        print("\n"+_nowDir)
        print("\n(2차 전처리 대상 파일이 존재하지 않습니다.)")
        sys.exit(1)

    for fname in f_list:
        if os.path.isdir(_nowDir + "/" + fname):
            dir_list.append(_nowDir + "/" + fname)
        elif os.path.isfile(_nowDir + "/" + fname):
            file_extension = os.path.splitext(fname)[1]
            if file_extension == ".csv" or file_extension == ".CSV":  # csv
                _filelist.append(_nowDir + "/" + fname)

    for toDir in dir_list:
        recursive_search_dir(toDir, _filelist)


@ray.remote
def preprocessing(csv_file, csv_path, out_path):

    skip_rows = 0
    total_lines = 0
    batch_size = 50000  # 500,000으로 설정 시, 메모리 부족 오류 발생 (해당 서버 정보: 코어개수 16, Ram: 8G, 사용 CPU 수: 5)
    isError=False
    cols = []
    while True:
        try:
            if len(cols) == 0:
                df = pd.read_csv(csv_file, low_memory=False, skiprows=skip_rows, nrows=batch_size)
            else:
                df = pd.read_csv(csv_file, low_memory=False, names=cols, header=None, skiprows=skip_rows, nrows=batch_size)

            for field in df.columns:
                if 'Unnamed: ' in field:
                    df.drop(columns = field, inplace=True)
        except Exception as e:
            if str(e).startswith('No columns to parse from file'):
                break
            else:
                print('\n')
                print(csv_file)
                print(e)
                isError=True
                break

        if len(df) == 0:
            break
        else:
            total_lines += len(df)
            skip_rows += batch_size

            # 필드명을 모두 소문자로 변환 (사유: 22년 6월 데이터부터, 필드명이 대문자로 바뀜. 이전 데이터와 통일하기 위해.)
            df.rename(mapper=str.lower, axis='columns', inplace=True)
        try:
            df.sort_values(by='coll_dt', inplace=True)
            df.reset_index(drop=True, inplace=True)
        except Exception as e:
            print('\n')
            print(csv_file)
            print(e)
            isError=True
            break
        cols = list(df.columns)

        # 날짜, 시간 형식에서 'T', 'Z' 문자 제거 ('T', 'Z' 포함되는 것은 ISO 8601 형태이며, 데이터 통일을 위해 문자 제거)
        df['coll_dt'] = df['coll_dt'].str.replace('T', ' ').str.replace('Z', '')

        # 만약 차종 필드가 없을 경우 (2021년 sk 파일), 새로 생성
        if 'car_type' not in df:
            try:
                # 차종 필드 새로 생성
                df['car_type'] = csv_file.split('/')[-2]
            except:
                None

        # 연식 필드 새로 생성
        # 기존 저장된 차량ID-연식 환산 정보가 있으면, 불러오기
        model_year_dict = {}
        try:
            with open('../split_csv_by_carid/carid_model_year.pickle', 'rb') as fr:
                model_year_dict = pickle.load(fr)
        except Exception as e:
            print(e)
            pass
        # 차량 연식 정보 불러오기 (있는 경우)
        if (df['dev_id'].iloc[0] in model_year_dict):
            df['car_model_year'] = model_year_dict[df['dev_id'].iloc[0]]
        else:
            df['car_model_year'] = float('nan')

        for i in range(len(df)):
            df_curr_row = df.iloc[i]

            # 오류 값 처리
            for field in df.columns:
                val = df_curr_row.loc[field]
        
                try:
                    val = float(val)
                except:
                    continue

                # soc 필드인 경우, 값이 0~100을 벗어나면 오류값이므로 NaN값으로 변경
                if 'soc' in field:
                    if val < 0 or val > 100:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # soh 필드인 경우, 값이 0~100을 벗어나면 오류값이므로 NaN값으로 변경
                if 'soh' in field:
                    if val < 0 or val > 100:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 전압 필드인 경우, 값이 3,000 초과이면 오류값으로 판단 NaN값으로 변경
                if '_volt' in field:
                    if val > 3000:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 온도 필드인 경우, 값이 -35 미만 또는 +80 초과이면 오류값으로 판단 NaN값으로 변경
                if '_temp' in field:
                    if val < -35 or val > 80:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 전류 필드인 경우, 값이 -500 미만 또는 +500 초과이면 오류값으로 판단 NaN값으로 변경
                if 'b_pack_current' in field:
                    if val < -500 or val > 500:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 차량 속도 필드인 경우, 값이 0 미만 또는 180 초과이면 오류값으로 판단 NaN값으로 변경
                if 'car_speed' in field:
                    if val < 0 or val > 180:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 배터리 전압 필드인 경우, 값이 0 미만 또는 6 초과이면 오류값으로 판단 NaN값으로 변경
                if field.startswith('b_cell') and field.endswith('_volt'):
                    if val < 1 or val > 6:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 누적 관련 양 값들이 1,000,000 초과이면 오류값으로 판단 NaN값으로 변경
                if 'b_accum' in field:
                    if val > 1000000:
                        df.loc[df.index[i], field] = np.nan
                    continue

                # 누적주행거리 필드인 경우, 값이 0 이하 또는 2,000,000 초과이면 오류값으로 판단 NaN값으로 변경
                if 'mileage' in field:
                    if val <= 0 or val > 2000000:
                        df.loc[df.index[i], field] = np.nan

        # csv 파일에 저장
        subpath = '/'.join(list((csv_file.split(csv_path)[-1]).split('/'))[:-1])
        if len(subpath) == 0:
            subpath = '.'
        result_path = out_path + '/' + subpath + '/'
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        file_name = csv_file.split('/')[-1]
        if os.path.isfile(result_path + file_name):
            df.to_csv(result_path + file_name, mode='a', header=False, index=False)
        else:
            df.to_csv(result_path + file_name, mode='w', header=True, index=False)
    if not isError and rmfile == 'rm':
        print ("removing original files: ",csv_file)
        os.remove(csv_file)
    return total_lines


if __name__ == '__main__':

    pn = int(sys.argv[1].replace('\r', ''))
    csv_path = sys.argv[2].replace('\r', '')
    out_path = sys.argv[3].replace('\r', '')
    rmfile = sys.argv[4]

    if pn <= 0:
        ray.init()
    else:
        ray.init(num_cpus=pn)

    if csv_path[-1] == '/':
        csv_path = csv_path[:-1]
    if out_path[-1] == '/':
        out_path = out_path[:-1]

    print (" CSV Dir location = ", csv_path)
    csv_list = []

    proc_start_time = time.time()

    print('\n======================================================')
    print('sk 렌트카 데이터 전처리 시작')
    print('======================================================')
    print('CSV 파일 목록 불러오는 중..')
    recursive_search_dir(csv_path, csv_list)

    total_csv_cnt = len(csv_list)

    print('총 CSV파일 수 : {}'.format(total_csv_cnt))

    obj_id_list = []
    print('\nCSV 파일 2차 전처리 중..')
    for csv_file in csv_list:
        obj_id_list.append(preprocessing.remote(csv_file, csv_path, out_path))

    cnt=1
    processed_lines=0
    result_path_list = set()

    while len(obj_id_list):
        printProgressBar(cnt, total_csv_cnt)
        done, obj_id_list = ray.wait(obj_id_list)
        processed_lines += ray.get(done[0])
        cnt+=1


    print("\n전처리된 데이터 라인 수 : {}".format(processed_lines))
    print('total running time : {:.2f} sec'.format(time.time()-proc_start_time))
