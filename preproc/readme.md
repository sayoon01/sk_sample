# preproc


## 개요(코드 설명)
- 차량ID 별로 분리된 CSV파일의 데이터 규격화 및 전처리 수행
- ray 모듈을 이용한 병렬 처리
- (참고: sk_2021 파일도 실행 가능)

## 코드 실행 방법

### 입력 데이터 다운로드
- 차량별 파일 분할된 파일 저장 위치 참고
  - [../split_csv_by_carid/readme.md](https://github.com/dongsikyoon/SW-Platform/blob/main/apps/preprocessing/sk_2022_preprocessing/split_csv_by_carid/readme.md)

- (서버에 저장)차량별로 파일이 분할된 파일 현재위치에 다운로드
  ```
  rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_2022_splited_by_carid .
  비밀번호 : evdataset1234!
  ```

### 실행 인자값 설명
- pn : 사용할 CPU 수 (0 입력시 모든 cpu 사용)
- file_input_path : file 입력 경로
- file_output_path : file 출력 경로
- rmfile : 입력파일 보존 : 'pre', 삭제 : 'rm'

### 실행방법
| 데이터 종류           | 데이터 저장 위치(서버)                                                                                                                                 | 설정인자값(info.ini)                                                                                       | 실행                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| EV- SK rent 2022 | /home/data/hdd_new/sk_2022_csv | pn=0<br>file_input_path='../../../../data_files/EV/sk_2022_splited_by_carid'<br>file_output_path='../../../../data_files/EV/sk_2022_preproced/'<br>rmfile='pre'                    | bash run.sh |


## 출력 데이터 다운로드
```
rsync -avhz -e 'ssh -p 9990' --progress --partial data@59.14.241.229:/home/data/hdd_new/sk_2022_preproced .
```


## 실행 결과
```
goo4168@tinyos-PEdge-R440:/hdd_lvm/goo4168/SW-Platform/apps/preprocessing/sk_2022_preprocessing/preproc
bash run.sh


>>==============================================================
실행 관련 주요 정보(run.sh)
file 입력 경로   : ../../../../../sk_2022_csv/
file 출력 경로   : ../../../../../sk_2022_preproced/
ID필드 명        : dev_id
==============================================================<<
 CSV Dir location =  ../../../../../sk_2022_csv

======================================================
CSV ID별 분류 시작
======================================================
CSV 파일 목록 불러오는 중..
총 CSV파일 수 : 699

CSV ID별 분류 중..
Progress |██████████████████████████████████████████████████| 100.0% Complete

총 처리한 CSV row 수 : 85064629
출력 파일 경로 : ../../../../../sk_2022_csv
total running time : 18734.94 sec
 *** end script run for PYTHON ***
```

---
# 추가사항 (4, 6, 7, 8월 데이터)
- 입력 데이터 
```
#4월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2204_split_by_carid_v2 .

#6월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2206_split_by_carid .

#7월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2207_split_by_carid .

#8월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2208_split_by_carid .
```

- 출력 데이터
```
#4월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2204_preproced_v2 .

#6월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2206_preproced .

#7월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2207_preproced .

#8월 데이터
rsync -avhz -e 'ssh -p 9595' --progress --partial sim@59.14.241.229:/home/sim/dataDisk/data/EV/sk_2208_preproced .
```
