# 日本語の指示で動く移動ロボット

Physical AI 応用1講座 最終課題。

日本語で「青いボールを確認してから、緑の箱の前で止まって」のように指示すると、LLM（Claude API）が行動計画を立て、MuJoCo 上の二輪ロボットが前方カメラの画像を頼りに目標を探し、近づき、止まります。Windows 上だけで動作します。

```
日本語の指示 → LLM（行動計画）→ 計画の検証 → 実行器
前方カメラ（色画像＋距離画像）→ 色と距離の認識 → 移動制御 → 二輪ロボット
```

- **計画**：LLM はツール呼び出し `submit_plan` で `search`（探索）・`approach`（接近）・`stop`（停止）の計画だけを返します。モーター値やコードは生成させません。
- **検証**：受け取った計画はプログラム側でも検査します（1〜8動作、対象は red / green / blue のみ、最後は stop）。不正な計画や失敗した動作があれば、残りの実行を中止します。
- **認識と制御**：カメラの色画像と距離画像だけで目標の方向と距離を求めます。物体の座標は制御に使いません。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

`.env` に `ANTHROPIC_API_KEY` を設定します。`ANTHROPIC_MODEL` は空欄なら `claude-opus-5-5` を使います。`.env` は Git の管理対象外です。描画には OpenGL に対応したグラフィックス環境が必要です。

## 使い方

### 日本語の指示で動かす

```powershell
.\.venv\Scripts\python.exe src\main.py --instruction "青いボールを確認してから、緑の箱の前で止まって"
.\.venv\Scripts\python.exe src\main.py --instruction "赤い箱へ行って" --plan-only
```

`--plan-only` は計画を表示するだけでロボットを動かしません（API は呼び出します）。`--headless` で画面なしで実行できます。実行ログと画像は `outputs/instruction/` に保存されます。

API を使わない確認用に `--planner mock` があります。文中の「赤・緑・青」を順に拾うだけの単純な規則で、AI は使いません。

### 手動操作

```powershell
.\.venv\Scripts\python.exe src\main.py
```

↑/↓ で前進・後退、←/→ で旋回、Space で停止、Backspace で初期位置に戻ります（英字キーは MuJoCo ビューアの表示切替に割り当てられているため使いません）。

### 色を指定して自動接近

```powershell
.\.venv\Scripts\python.exe src\main.py --target green
```

見えなければその場で回転して探し、見えたら画像の中央に合わせて近づき、目標面まで約 0.65 m で止まります。

### 録画と提出用動画

```powershell
.\.venv\Scripts\python.exe src\main.py --instruction "青いボールを確認してから、緑の箱の前で止まって" --headless --record outputs/demo.mp4
.\.venv\Scripts\python.exe scripts\build_video.py --demo outputs/demo.mp4 --output outputs/submission.mp4
```

1つ目は実行の様子を MP4（1280×720、10fps、通常速度）に保存し、指示・計画・実行中の手順・前方カメラ映像を重ねて表示します。2つ目は説明スライドを加えて1〜3分の動画にまとめます。既存のファイルは上書きしません。

## テスト

```powershell
.\.venv\Scripts\python.exe src\main.py --check
.\.venv\Scripts\python.exe src\main.py --check-autonomy
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

順に、移動とカメラ描画の確認、3色それぞれへの探索・接近の確認（目標座標は合否判定にだけ使用）、計画の検証と実行中止の単体テストです。いずれも API は呼び出しません。

## 構成

| パス | 内容 |
|---|---|
| `assets/room.xml` | 部屋・目標・二輪ロボット・前方カメラ（MJCF） |
| `src/sim.py` | シミュレーションと二輪の速度制御 |
| `src/perception.py` | 色画像と距離画像からの目標認識 |
| `src/autonomy.py` | 探索・接近の制御 |
| `src/planner.py` | LLM による行動計画と計画の検証 |
| `src/executor.py` | 計画の順次実行 |
| `src/recording.py` | MP4 録画 |
| `scripts/build_video.py` | 説明スライドを加えた提出用動画の作成 |
| `docs/report.html` | レポート |

## 制限

各色1個の目標と、障害物のない部屋を前提にしています。同じ色の物体の区別、形の認識、障害物回避、実行結果に基づく再計画には対応していません。
