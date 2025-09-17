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
    python3 sk_preprocessing.py $pn $file_input_path $file_output_path $rmfile
    echo " *** end script run for PYTHON *** "
    exit 0 #finish successfully
else

    #---------------------- 필수 입력 인자 ----------------------
    # [1] 사용할 CPU 수
    pn=0
    # [2] file 입력 경로
    file_input_path='../../../../../sk_2022_splited_by_carid/'
    #file_input_path='/mnt/hdd/sk_2022_splited_by_carid/'
    # [3] file 출력 경로
    file_output_path='../../../../../sk_2022_preproced/'
    #file_output_path='/mnt/hdd/sk_2022_preproced/'
    rmfile='pre' #입력파일 보존, 삭제하지 않음 'rm'의 경우 삭제

    #---------------------- 필요에따라 추가할 입력 인자 ----------------------

    if [ -d $file_input_path ]; then
        echo ''
    else
        echo '입력하신 file input path 디렉토리가 존재하지 않습니다.'
        echo '프로그램 종료.'
        exit 100
    fi


    echo ">>===================================================="
    echo "실행 관련 주요 정보(this_run.sh)"
    echo "사용할 CPU 수    : "$pn
    echo "전처리할 csv dir path   : "$file_input_path
    echo "file 출력 경로   : "$file_output_path
    echo "입력파일 삭제 여부   : "$rmfile
    echo "====================================================<<"

    python3 sk_preprocessing.py $pn $file_input_path $file_output_path $rmfile
    echo " *** end script run for PYTHON *** "
fi
