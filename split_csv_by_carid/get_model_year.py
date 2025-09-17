# -*- coding:utf-8 -*-

import os
import pandas as pd
import sys
import pickle

sys.path.append('../../../minIO')
from MinioData import MinioData

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
        print("\n(CSV ID별 분류 대상 파일이 존재하지 않습니다.)")
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

if __name__ == '__main__':

    #_input_bucket = sys.argv[1]
    _input_path = sys.argv[1]
    _id = sys.argv[2].replace('\r', '')
    _vin = sys.argv[3].replace('\r', '')
    if _input_path[-1] == '/':
        _input_path = _input_path[:-1]
    _output_path = sys.argv[4]

    print (" CSV Dir location = ", _input_path)
    csv_list = []

    print('======================================================')
    print('CSV 파일 목록 불러오는 중..')
    recursive_search_dir(_input_path, csv_list)
    csv_len = len(csv_list)

    if csv_len == 0:
        exit('csv 파일이 없습니다. 프로그램 종료.')
    print('총 csv파일 수 : {}'.format(csv_len))

    # 차대번호 10번째 자리를, 연식으로 환산하는 표
    vin_model_year_convertion_table = {
        '5': 2005,
        '6': 2006,
        '7': 2007,
        '8': 2008,
        '9': 2009,
        'A': 2010,
        'B': 2011,
        'C': 2012,
        'D': 2013,
        'E': 2014,
        'F': 2015,
        'G': 2016,
        'H': 2017,
        'J': 2018,
        'K': 2019,
        'L': 2020,
        'M': 2021,
        'N': 2022,
        'P': 2023,
        'R': 2024,
        'S': 2025,
        'T': 2026,
        'V': 2027,
        'W': 2028,
        'X': 2029,
        'Y': 2030,
        '1': 2031,
        '2': 2032,
        '3': 2033,
        '4': 2034,
        '5': 2035,
        '6': 2036,
        '7': 2037,
        '8': 2038,
        '9': 2039
    }

    existing_model_year_dict = {}
    added_model_year_dict = {}

    # 기존 저장된 ID-연식 환산 정보가 있으면, 불러오기
    try:
        with open(_output_path, 'rb') as fr:
            existing_model_year_dict = pickle.load(fr)
    except Exception as e:
        print(e)
        pass

    print('\n차량별 연식 추출 중..')
    cnt = 1
    added_num = 0

    for csv_file in csv_list:
        printProgressBar(cnt, csv_len)
        temp_cnt = 1

        skip_rows = 0
        batch_size = 500000

        while True:
            try:
                df = pd.read_csv(csv_file, low_memory=False, usecols=[_id, _vin], skiprows=range(1, skip_rows), nrows=batch_size)

            except Exception as e:
                if str(e).startswith('Usecols do not match columns'):
                    skip_rows += batch_size
                    continue
                elif str(e).startswith('No columns to parse from file'):
                    print('\n')
                    print(csv_file)
                    print(e)
                    exit()

            if len(df) == 0:
                break
            else:
                # 차대번호가 null이면 제외
                tdf = df.query("%s.notna()" %(_vin))
                tdf = tdf.drop_duplicates()
                id_list = tdf[_id].unique()

                for i in range(len(tdf)):
                    if (tdf[_id].iloc[i] in existing_model_year_dict) or (tdf[_id].iloc[i] in added_model_year_dict):
                        continue
                    else:
                        # 차대번호 중 10번째 자리 값으로, 연식 추출
                        model_year = vin_model_year_convertion_table[tdf[_vin].iloc[i][9]]
                        added_model_year_dict[tdf[_id].iloc[i]] = model_year
                        added_num += 1

                print(str(temp_cnt * batch_size) + '행 처리 완료')
                temp_cnt += 1

                skip_rows += batch_size

    # 새로 추가된 ID-연식 환산 정보를, 기존 ID-연식 환산 정보와 합쳐서, 출력 파일에 덮어서 저장
    with open(_output_path, 'wb') as fw:
        existing_model_year_dict.update(added_model_year_dict)
        pickle.dump(existing_model_year_dict, fw)

    print(added_model_year_dict)
    print('\n새로 추가된 차량 연식 정보 수: ' + str(added_num))
    print('\n차량별 연식 추출 및 저장 완료(' + _output_path + ')')