import os, sys
import pandas as pd

def process_one(in_path, out_path):
    try:
        df = pd.read_csv(in_path, low_memory=False)
    except Exception as e:
        print(f"[READ-FAIL] {in_path} -> {e}")
        return 0
    # 컬럼 소문자
    df.columns = [c.lower().strip() for c in df.columns]
    # coll_dt 없으면 스킵(목록/메타 파일)
    if 'coll_dt' not in df.columns:
        print(f"[SKIP] {in_path} : 'coll_dt' 없음")
        return 0
    # 시간 파싱 + 정렬
    df['coll_dt'] = pd.to_datetime(df['coll_dt'], errors='coerce')
    df = df.dropna(subset=['coll_dt']).sort_values('coll_dt').reset_index(drop=True)
    # 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] {in_path} -> {out_path}  (rows={len(df)})")
    return len(df)

def main(indir, outdir):
    total = 0
    for root, _, files in os.walk(indir):
        for f in files:
            if not f.lower().endswith(".csv"): 
                continue
            src = os.path.join(root, f)
            # 하위 폴더 구조 유지
            rel = os.path.relpath(src, indir)
            dst = os.path.join(outdir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            total += process_one(src, dst)
    print(f"\n=== DONE: total rows written = {total} ===")

if __name__ == "__main__":
    indir = sys.argv[1] if len(sys.argv) > 1 else "./raw_csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "./clean_csv"
    main(indir, outdir)
