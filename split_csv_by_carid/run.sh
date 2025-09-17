#!/bin/bash

# 파이션 코드 실행스크립트

if [ -f 'info.ini' ]; then
    source info.ini
    echo ""
    echo "using information in the info.ini file"
    echo ""
    echo ">>=============================================================="
    echo "실행 관련 주요 정보(run.sh)"
    echo "사용할 CPU 수    : "$pn
    echo "file 입력 경로   : "$file_input_path
    echo "file 출력 경로   : "$file_output_path
    echo "ID필드 명        : "$id
    echo "==============================================================<<"
    python3 split_csv_by_carid.py $pn $file_input_path $file_output_path $id
    exit 0 #finish successfully
else
    #---------------------- 필수 입력 인자 ----------------------
    # [1] 사용할 CPU 수
    pn=0
    # [2] file 입력 경로
    file_input_path='../../../../../sk_2022_data/'
    # [3] file 출력 경로
    file_output_path='../../../../../sk_2022_splited_by_carid/'
    # [4] id 필드명
    id='dev_id'
    #---------------------- 필요에따라 추가할 입력 인자 ----------------------

    if [ -d $file_input_path ]; then
        echo ''
    else
        echo '입력하신 file input path 디렉토리가 존재하지 않습니다.'
        echo '프로그램 종료.'
        exit 100
    fi


    echo ""
    echo ">>=============================================================="
    echo "실행 관련 주요 정보(run.sh)"
    echo "사용할 CPU 수    : "$pn
    echo "file 입력 경로   : "$file_input_path
    echo "file 출력 경로   : "$file_output_path
    echo "ID필드 명        : "$id
    echo "==============================================================<<"


    python3 split_csv_by_carid.py $pn $file_input_path $file_output_path $id

    echo " *** end script run for PYTHON *** "
fi
