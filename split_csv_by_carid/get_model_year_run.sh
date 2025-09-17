#!/bin/bash

# 파이션 코드 실행스크립트

#---------------------- 필수 입력 인자 ----------------------
# [1] file 입력 경로 (2211, 2212, 2307-2310까지 했음)
input_path='/mnt/hdd/sk_origin/sk_2211_origin'
# [2] id 필드명
id='DEV_ID'
# [3] 차대번호 필드명
vin='VIN'
# [4] file 출력 경로
output_path='./carid_model_year.pickle'

echo ""
echo ">>=============================================================="
echo "실행 관련 주요 정보(run.sh)"
echo "csv 입력 경로 : "$input_path
echo "ID 필드명     : "$id
echo "차대번호 필드명: "$vin
echo "file 출력 경로: "$output_path
echo "==============================================================<<"

python3 get_model_year.py $input_path $id $vin $output_path
exit 0 #finish successfully