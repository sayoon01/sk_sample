# split_csv_by_carid


## 개요(코드설명)
- sk 2022 원본데이터는 차종/날짜 별로 파일이 분리되어 있어 한 CSV파일에 여러 차량의 데이터가 존재함
- CSV파일을 읽어 각 차량 ID별로 데이터를 추출하여 CSV 파일로 저장
- 그외 원본데이터의 필드가 361개인데(마지막 필드 23개는 오류값) 유효한 필드 338개만 추출하여 오류 필드 제외시킴
- ray 모듈을 이용한 병렬 처리

## 코드 실행 방법

### 입력 데이터 다운로드
- 원본 데이터파일 다운로드 코드 및 위치는 아래 링크 참조
  - [transfer_datafiles](https://github.com/dongsikyoon/SW-Platform/tree/main/apps/csv_collection/transfer_datafiles)

- 원본 데이터파일 현재위치에 다운로드
  ```
  rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_2022_data .
  비밀번호 : evdataset1234!
  ```

### 실행 인자값 설명
- pn : 사용할 CPU 수 (0 입력시 모든 cpu 사용)
- file_input_path : file 입력 경로
- file_output_path : file 출력 경로
- id : 차량 번호 필드명

### 실행방법
| 데이터 종류           | 데이터 저장 위치(서버)                                                                                                                                 | 설정인자값(info.ini)                                                                                       | 실행                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| EV- SK rent 2022 | /home/data/hdd_new/sk_2022_data | pn=0<br>file_input_path='../../../../data_files/EV/sk_2022_data/'<br>file_output_path='../../../../data_files/EV/sk_2022_splited_by_carid/'<br>id='dev_id'                    | bash run.sh |


## 출력 데이터 다운로드(pw:evdataset1234!)
```
rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_2022_splited_by_carid .
```


### 실행 결과
```
goo4168@tinyos-PEdge-R440:/hdd_lvm/goo4168/SW-Platform/apps/preprocessing/sk_2022_preprocessing/split_csv_by_carid$
bash run.sh


>>==============================================================
실행 관련 주요 정보(run.sh)
file 입력 경로   : ../../../../../sk_2022_data/
file 출력 경로   : ../../../../../sk_2022_csv/
ID필드 명        : dev_id
==============================================================<<
 CSV Dir location =  ../../../../../sk_2022_data

======================================================
CSV ID별 분류 시작
======================================================
CSV 파일 목록 불러오는 중..
총 CSV파일 수 : 240

CSV ID별 분류 중..
Progress |██████████████████████████████████████████████████| 100.0% Complete

총 처리한 CSV row 수 : 85064629
출력 파일 경로 : ../../../../../sk_2022_csv
total running time : 7378.89 sec
 *** end script run for PYTHON ***
```


---
# 추가사항

## 개요
- 기존 데이터와는 달리 2022년 4월, 6월, 7월, 8월 데이터는 차종에 대한 정보가 없음
- 기존 SK데이터에서 차량 ID별 차종 데이터를 CSV파일로 추출한 것을 가지고 현재 차량 ID와 비교하여 차종정보를 기입

## 코드 실행 방법

### 입력 데이터 다운로드 
- 원본 데이터파일 현재위치에 다운로드
  ```
  #4월 데이터
  rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_data_2204 ./
  
  #6월 데이터
  rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_data_2206 ./
  
  #7월 데이터
  rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_data_2207 ./

  #8월 데이터
  rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_data_2208 ./
  ```

### 출력 데이터 다운로드(pw:keti1234!)
- 원본 데이터파일 현재위치에 다운로드
```
#4월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2204_split_by_carid ./

#6월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2206_split_by_carid ./

#7월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2207_split_by_carid ./

#8월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2208_split_by_carid ./
```

### 실행 결과
```
sim@sim-desktop:~/dataDisk/SW-Platform/apps/preprocessing/sk_2022_preprocessing/split_csv_by_carid$ bash run.sh 

using information in the info.ini file

>>==============================================================
실행 관련 주요 정보(run.sh)
사용할 CPU 수    : 8
file 입력 경로   : ./../../../../../data/sk_data_2206
file 출력 경로   : ./../../../../../data/EV/sk_2206_split_by_carid
ID필드 명        : DEV_ID
==============================================================<<
 CSV Dir location =  ./../../../../../data/sk_data_2206

======================================================
CSV ID별 분류 시작
======================================================
CSV 파일 목록 불러오는 중..
총 CSV파일 수 : 1

CSV ID별 분류 중..
Progress |██████████████████████████████████████████████████| 100.0% Complete

총 처리한 CSV row 수 : 23921039
출력 파일 경로 : ./../../../../../data/EV/sk_2206_split_by_carid
total running time : 9742.39 sec
 *** end script run for PYTHON ***
```
