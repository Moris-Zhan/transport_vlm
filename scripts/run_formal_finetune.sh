#!/bin/bash
# run_formal_finetune.sh — 外部監督迴圈：這台機器已知會在隨機 step 數發生硬體/驅動層級的
# SIGSEGV/SIGKILL 中斷（RTX 5090 / Blackwell 驅動，見 fine_tuning_vlm_trl_traffic_tune_ministral.py
# 的說明），訓練腳本本身只負責「偵測既有 checkpoint、自動從那裡續跑」，實際「崩潰後重新啟動」
# 這件事一直沒有腳本落地，都是手動重跑。這支腳本把重跑這件事自動化。
#
# 用法：
#   ./run_formal_finetune.sh [傳給訓練腳本的參數...]
#   例：./run_formal_finetune.sh --save_steps 100


# ./scripts/run_formal_finetune.sh --model_id Ministral-3-8B-Instruct-2512 --num_train_epochs 10 \
# --weight_decay 0.01 --eval_strategy steps --eval_steps 100 \
# --data_root data/collision_diagram/synthetic_tune_simple_v2 \
# --early_stopping_patience 10 --new_run

#
# 環境變數：
#   FINETUNE_SCRIPT   要跑的訓練腳本，預設 fine_tuning_vlm_trl_traffic_tune_ministral.py
#   FINETUNE_PYTHON   python 直譯器路徑，預設 /root/miniconda3/envs/fine_tune_vlm/bin/python
#   FINETUNE_LOG      訓練日誌路徑，預設 formal_finetune_ministral.log（跟 monitor_finetune.sh
#                     的 FINETUNE_LOG 用同一個變數名稱，方便兩支腳本指到同一個檔案；預設值改成
#                     跟腳本名稱綁在一起，避免跟其他模型的正式訓練共用同一個 formal_finetune.log
#                     混在一起，難以分辨是哪次訓練的紀錄——實測發生過）。
#   MAX_CONSECUTIVE_FAILURES  連續失敗幾次就放棄、不再重試（避免真的是程式 bug 時無限重跑
#                             一直炸），預設 20
#   FAILURE_TAIL_LINES        訓練腳本某次 attempt 失敗時，印到終端機的錯誤輸出行數（從該次
#                             attempt 輸出的尾端算起），預設 200。之所以只印尾端而不是整段
#                             輸出，是因為訓練過程的 tqdm 進度條輸出量很大，全印出來終端機會
#                             被洗掉、反而看不到真正的錯誤訊息；Python traceback 一定出現在
#                             輸出最後面，印尾端幾百行就夠看到完整錯誤。完整輸出不受影響，一律
#                             照舊寫進 FINETUNE_LOG。
#   STATUS_INTERVAL_SECONDS   訓練「進行中」時，每隔幾秒印一次目前的訓練速度（秒/step），
#                             預設 300（5 分鐘）。一次 attempt 動輒跑好幾小時，只在 attempt
#                             結束時印一次會讓這行資訊在終端機上一閃即逝、隔天回來看只剩最後
#                             一次；改成訓練還在跑的時候就定期印，同一行文字會在 attempt 執行
#                             期間反覆出現，才算「長時間顯示」。
#   STALL_TIMEOUT_SECONDS     訓練 process 還活著、但 tqdm 進度條的 step 數連續這麼多秒完全
#                             沒有前進，就視為卡死（懷疑是 GPU 驅動層級 wedge：訓練腳本裡的
#                             torch.cuda.synchronize() 沒有 timeout，一旦驅動真的卡住，
#                             process 不會結束、也不會有任何新輸出），強制送 SIGKILL 給整個
#                             process group 並照失敗流程重試，避免監督迴圈陪著一起卡死到
#                             天荒地老。預設 1800（30 分鐘）。設為 0 則關閉這項偵測。
#   LIVE_PROGRESS_INTERVAL_SECONDS  訓練「進行中」時，終端機上即時進度那一行多久刷新一次
#                             （用 \r 蓋掉前一次、不佔用捲動記錄，也不寫進 FINETUNE_LOG——
#                             FINETUNE_LOG 已經有 STATUS_INTERVAL_SECONDS 那組正式摘要）。
#                             只在真的連著終端機（不是被導向檔案/管線）時才會顯示。預設 10
#                             （秒）。設為 0 則關閉這項即時刷新，只保留原本每
#                             STATUS_INTERVAL_SECONDS 秒印一次的摘要。
#
# 訓練還在跑的時候，每隔 STATUS_INTERVAL_SECONDS 秒，以及 attempt 結束（不論正常結束或中斷）
# 時，都會從目前為止的輸出裡的 tqdm 進度條算出「平均每 step 花幾秒」並印出來（同時寫進終端機
# 和 FINETUNE_LOG），方便觀察訓練速度有沒有異常變慢（例如 GPU 被其他程序搶用）、或拿來估算
# 剩餘時間。
#
# 若沒有傳任何參數給訓練腳本，預設會加上 --model_id Ministral-3-8B-Instruct-2512（因為
# FINETUNE_SCRIPT 預設就是 Ministral 專用的腳本，而 --model_id 本身的預設值是共用設定裡的
# GLM-4.1V-9B-Thinking，兩者對不上時腳本會直接報錯拒絕啟動——實測發生過：沒帶
# --model_id 直接跑，會在 3 秒內就失敗、被這支監督迴圈誤判成硬體隨機中斷，連續重試到
# MAX_CONSECUTIVE_FAILURES 才停下來）。想跑其他模型，自己在後面帶完整的
# --model_id <值> 就會蓋掉這個預設。

set -uo pipefail

# 這支腳本現在放在 scripts/ 底下，自身所在目錄不再等於專案根目錄，所以往上一層
# 才是真正的 PROJECT_DIR（Output/、models/、FINETUNE_LOG 等都要相對於這裡解析）。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py}"
FINETUNE_PYTHON="${FINETUNE_PYTHON:-/root/miniconda3/envs/Mistral3-fa-vllm/bin/python}"
FINETUNE_LOG="${FINETUNE_LOG:-formal_finetune_ministral.log}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-20}"
FAILURE_TAIL_LINES="${FAILURE_TAIL_LINES:-200}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-300}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1800}"
LIVE_PROGRESS_INTERVAL_SECONDS="${LIVE_PROGRESS_INTERVAL_SECONDS:-10}"
LOG_FILE="$PROJECT_DIR/$FINETUNE_LOG"

# 從輸出檔（目前為止累積的、或整個 attempt 結束後完整的都可以）裡的 tqdm 進度條算出
# 「平均每 step 花幾秒」並印出來（同時寫進終端機和 FINETUNE_LOG）。tqdm 輸出到非終端機
# （這裡是被導向暫存檔）時，同一顆進度條每次更新是用 \r 蓋掉前一行、不是換行，所以要先把
# \r 轉成 \n 才能用一般文字工具抓到每一次更新；取最後一行就是目前為止最新的速度數字。
# 進度條可能印成「X.XXit/s」（每秒幾個 step）或「X.XXs/it」（每個 step 幾秒），兩種都要
# 換算成統一的「秒/step」再印出來。抓不到任何進度條（例如訓練才剛啟動、還沒印出第一次更新）
# 就什麼都不印，靜靜等下一次再檢查。
print_step_speed() {
    local src_file="$1"
    local label="$2"
    local step_progress_line rate_str rate_value step_seconds

    step_progress_line=$(tr '\r' '\n' < "$src_file" | grep -E '%\|.*(it/s|s/it)' | tail -1)
    if [ -z "$step_progress_line" ]; then
        return
    fi

    rate_str=$(printf '%s' "$step_progress_line" | grep -oE '[0-9.]+(it/s|s/it)' | tail -1)
    rate_value=$(printf '%s' "$rate_str" | grep -oE '^[0-9.]+')
    if printf '%s' "$rate_str" | grep -q 'it/s$'; then
        step_seconds=$(awk -v v="$rate_value" 'BEGIN { if (v > 0) { printf "%.3f", 1 / v } else { print "NA" } }')
    else
        step_seconds="$rate_value"
    fi
    _flush_live_progress_line
    echo "=== [supervisor] ${label}訓練速度：平均約 ${step_seconds} 秒/step（依最後一筆進度：${rate_str}）$(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_FILE"
}

# 即時（每 LIVE_PROGRESS_INTERVAL_SECONDS 秒）在終端機上刷新目前的 tqdm 進度：只抓最新一次
# 更新，用 \r 蓋掉同一行，不換行、不寫進 log（log 已經有 print_step_speed 每
# STATUS_INTERVAL_SECONDS 秒寫一次的正式摘要，這裡單純是給正在盯著終端機的人看的即時畫面）。
# 只在真的連著終端機時才印（[ -t 1 ]），避免污染被導向檔案/管線的輸出、也避免控制字元混進
# log。LIVE_PROGRESS_LINE_PRINTED 記錄「目前終端機上是不是停在一行還沒換行的即時進度」，讓
# 呼叫端（印正式摘要、印卡死警告、迴圈結束時）知道要不要先補一個換行，才不會把下一則訊息接在
# 同一行後面變得難以閱讀。
LIVE_PROGRESS_LINE_PRINTED=0
print_live_progress_line() {
    local src_file="$1" latest_line
    [ -t 1 ] || return
    [ "$LIVE_PROGRESS_INTERVAL_SECONDS" -gt 0 ] 2>/dev/null || return
    latest_line=$(tr '\r' '\n' < "$src_file" | grep -E '%\|' | tail -1)
    [ -z "$latest_line" ] && return
    # \033[K 清掉這一行游標之後的殘留字元（例如上一次進度字串比這次長）。
    printf '\r\033[K[即時進度] %s' "$latest_line"
    LIVE_PROGRESS_LINE_PRINTED=1
}

# 若目前終端機停在即時進度那一行（還沒換行），先補一個換行再印別的訊息，避免疊在同一行。
_flush_live_progress_line() {
    if [ "$LIVE_PROGRESS_LINE_PRINTED" -eq 1 ]; then
        printf '\n'
        LIVE_PROGRESS_LINE_PRINTED=0
    fi
}

# 從輸出檔裡最新一次 tqdm 進度條更新中，抓出目前的 step 數（例如
# 「45%|████ | 450/1000 [00:10<00:12, 13.05s/it]」抓出 450）。用來偵測「process 還活著、
# 但 step 數完全沒有前進」的卡死情況——print_step_speed() 抓的是速率，抓不到這件事（GPU
# 驅動 wedge 時，最後一次印出的速率會一直原地不動，但不代表訓練真的在動）。抓不到任何
# 進度條就回傳空字串，呼叫端要自行處理「目前還不知道 step 數」的情況。
get_current_step() {
    local src_file="$1"
    tr '\r' '\n' < "$src_file" | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | grep -oE '^[0-9]+'
}

if [ "$#" -eq 0 ]; then
    set -- --model_id Mistral-Small-3.2-24B-Instruct-2506
fi

attempt=0
consecutive_failures=0
# 一次 attempt 如果跑了至少這麼久才死掉，視為「有實際訓練到、只是被隨機硬體中斷」，把連續
# 失敗計數歸零；如果每次一啟動很快就死（例如參數打錯、程式真的有 bug），計數會一直往上疊，
# 到 MAX_CONSECUTIVE_FAILURES 就放棄，不會無限重跑一個註定失敗的指令。
MIN_PROGRESS_SECONDS=120

# 每次 attempt 的輸出先寫進這個暫存檔，結束後才 cat 進 FINETUNE_LOG（維持跟以前一樣的累積
# 日誌），失敗時再額外把尾端印到終端機——見上面 FAILURE_TAIL_LINES 的說明。用暫存檔而不是
# tee 即時雙印，是因為訓練成功時終端機不需要被整串 tqdm 進度條洗版。
ATTEMPT_OUT="$(mktemp)"
trap 'rm -f "$ATTEMPT_OUT"' EXIT

# Ctrl+C（SIGINT）/ 外部 kill（SIGTERM）處理：下面啟動訓練時用 setsid 讓訓練 process 變成
# 自己獨立的 session（脫離這支腳本所在終端機的 controlling tty），這代表終端機送出的 SIGINT
# 「只會打到這支監督腳本自己」，不會自動傳到已經脫離 session 的訓練 process——沒有這個 trap
# 的話，使用者按 Ctrl+C 只會中斷監督腳本，訓練 process 變成孤兒繼續留在背景默默跑，看起來
# 像是停掉了，其實沒有（實測發生過）。另外 bash 有個經典陷阱：下面 `wait -n ... || true`
# 這種寫法裡，收到的 SIGINT 常常會被 wait 提早返回、又被 `|| true` 吞掉，整個迴圈完全感覺不
# 到自己被中斷過，實測直接按 Ctrl+C 完全沒反應、監督迴圈甚至還會繼續開下一個 attempt。這裡
# 明確 trap INT/TERM，強制：印訊息 → 送 SIGTERM 給訓練 process 整個 group（給它機會正常收尾）
# → 等最多 5 秒 → 還沒死就 SIGKILL → exit，徹底跳出最外層 while true，不會再重試。
_shutting_down=0
_handle_interrupt() {
    # 避免使用者連續按好幾次 Ctrl+C 時重複跑一遍下面的清理邏輯。
    [ "$_shutting_down" -eq 1 ] && return
    _shutting_down=1
    _flush_live_progress_line
    echo "=== [supervisor] 收到中止訊號，終止訓練 process 並結束監督迴圈（不會再重試）$(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_FILE" >&2
    if [ -n "${train_pid:-}" ] && kill -0 "$train_pid" 2>/dev/null; then
        kill -TERM -- -"$train_pid" 2>/dev/null || kill -TERM "$train_pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$train_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$train_pid" 2>/dev/null; then
            echo "=== [supervisor] 訓練 process 5 秒內沒有回應 SIGTERM，改送 SIGKILL $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_FILE" >&2
            kill -KILL -- -"$train_pid" 2>/dev/null || kill -KILL "$train_pid" 2>/dev/null || true
        fi
    fi
    exit 130
}
trap '_handle_interrupt' INT TERM

while true; do
    attempt=$((attempt + 1))
    echo "=== [supervisor] attempt ${attempt}（連續失敗 ${consecutive_failures} 次）$(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

    start_ts=$SECONDS
    # 用 setsid 讓訓練 process 成為自己獨立的 process group leader（而不是沿用這支監督腳本
    # 的 group）：下面偵測到卡死、需要強制 kill 整個 group（訓練 process + 它開的 DataLoader
    # worker）時，才能用 `kill -- -"$train_pid"` 精準只殺這個 group，不會不小心殺到監督腳本
    # 自己（這支腳本沒有開 `set -m`，背景工作預設不會自動獲得獨立 process group）。
    setsid "$FINETUNE_PYTHON" "$FINETUNE_SCRIPT" "$@" > "$ATTEMPT_OUT" 2>&1 &
    train_pid=$!
    last_progress_step=""
    last_progress_ts=$SECONDS
    last_status_log_ts=$SECONDS

    # 輪詢間隔：預設用 LIVE_PROGRESS_INTERVAL_SECONDS（較短，例如 10 秒）當一次 tick，這樣
    # 終端機上的即時進度才能常常刷新；LIVE_PROGRESS_INTERVAL_SECONDS=0（關閉即時刷新）時退回
    # 用 STATUS_INTERVAL_SECONDS 當 tick，等同這個功能加入之前的輪詢頻率。
    poll_tick_seconds="$STATUS_INTERVAL_SECONDS"
    if [ "$LIVE_PROGRESS_INTERVAL_SECONDS" -gt 0 ] 2>/dev/null; then
        poll_tick_seconds="$LIVE_PROGRESS_INTERVAL_SECONDS"
    fi

    # 用「sleep 背景工作 race 訓練程序」而不是單純 `sleep N; check` 輪詢，是因為後者在訓練
    # 程序早早失敗時（例如參數打錯、3 秒內就死掉那種案例，見上面 MAX_CONSECUTIVE_FAILURES
    # 說明）會讓監督迴圈也跟著卡上一整個 poll_tick_seconds 才發現，等於把「快速失敗」誤判成
    # 「訓練中」；`wait -n` 只要訓練程序一結束就會立刻醒來，不受 sleep 剩餘時間影響，偵測失敗
    # 的延遲跟沒有這個定期印狀態功能之前一樣快（需要 bash >= 5.1 才支援 `wait -n <pid...>`
    # 指定特定 job 的寫法）。
    while kill -0 "$train_pid" 2>/dev/null; do
        sleep "$poll_tick_seconds" &
        status_sleep_pid=$!
        wait -n "$train_pid" "$status_sleep_pid" 2>/dev/null || true
        if kill -0 "$train_pid" 2>/dev/null; then
            # 醒來時訓練程序還活著，代表是 sleep 先到時間、不是訓練程序結束。
            # 每個 tick 都刷新一次終端機上的即時進度（見 print_live_progress_line 說明；
            # 只在真的連著終端機時才會實際印出東西）。
            print_live_progress_line "$ATTEMPT_OUT"

            # 累積滿 STATUS_INTERVAL_SECONDS 才印一次「正式」的平均速度摘要，同時寫進終端機
            # （換行）和 FINETUNE_LOG，維持這個功能原本的頻率與行為不變，不會因為 tick 變密
            # 集而洗版 log。
            if [ $((SECONDS - last_status_log_ts)) -ge "$STATUS_INTERVAL_SECONDS" ]; then
                print_step_speed "$ATTEMPT_OUT" "attempt ${attempt} 進行中，"
                last_status_log_ts=$SECONDS
            fi

            # 卡死偵測：process 活著不代表訓練真的有在動——這台機器已知有 GPU 驅動層級 wedge
            # 疑慮，腳本裡的 torch.cuda.synchronize() 沒有 timeout，一旦卡住，process 會停在
            # 原地不動也不會有任何新輸出。這裡比對 tqdm 目前的 step 數，STALL_TIMEOUT_SECONDS
            # 秒內完全沒有前進就強制中止，讓監督迴圈照失敗流程重試，而不是陪著一起卡死。
            if [ "$STALL_TIMEOUT_SECONDS" -gt 0 ]; then
                current_step=$(get_current_step "$ATTEMPT_OUT")
                if [ -n "$current_step" ] && [ "$current_step" != "$last_progress_step" ]; then
                    last_progress_step="$current_step"
                    last_progress_ts=$SECONDS
                elif [ $((SECONDS - last_progress_ts)) -ge "$STALL_TIMEOUT_SECONDS" ]; then
                    _flush_live_progress_line
                    echo "=== [supervisor] attempt ${attempt} 疑似卡死：step 數 ${STALL_TIMEOUT_SECONDS} 秒內沒有前進（目前 step=${current_step:-未知}），強制中止並重試 $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_FILE"
                    kill -KILL -- -"$train_pid" 2>/dev/null || kill -KILL "$train_pid" 2>/dev/null || true
                fi
            fi
        else
            # 訓練程序已經結束，把還沒到時間的 sleep 背景工作收掉，避免留下殭屍/多餘等待。
            kill "$status_sleep_pid" 2>/dev/null || true
            wait "$status_sleep_pid" 2>/dev/null || true
        fi
    done
    _flush_live_progress_line

    wait "$train_pid"
    exit_code=$?
    elapsed=$((SECONDS - start_ts))
    cat "$ATTEMPT_OUT" >> "$LOG_FILE"

    print_step_speed "$ATTEMPT_OUT" "attempt ${attempt} 結束，"

    if [ "$exit_code" -eq 0 ]; then
        echo "=== [supervisor] 訓練正常結束（exit 0），停止監督迴圈 $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"
        echo "訓練完成"
        exit 0
    fi

    echo "=== [supervisor] attempt ${attempt} 失敗（exit code=${exit_code}），錯誤輸出最後 ${FAILURE_TAIL_LINES} 行如下（完整內容見 $LOG_FILE） ===" >&2
    tail -n "$FAILURE_TAIL_LINES" "$ATTEMPT_OUT" >&2

    if [ "$elapsed" -ge "$MIN_PROGRESS_SECONDS" ]; then
        consecutive_failures=0
    else
        consecutive_failures=$((consecutive_failures + 1))
    fi
    echo "=== [supervisor] 訓練中斷，exit code=${exit_code}，跑了 ${elapsed} 秒（連續快速失敗 ${consecutive_failures} 次）$(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

    if [ "$consecutive_failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
        echo "=== [supervisor] 每次啟動都很快失敗、連續 ${MAX_CONSECUTIVE_FAILURES} 次，判斷不是隨機硬體中斷、可能是真的錯誤，停止重試 $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"
        echo "連續快速失敗 ${MAX_CONSECUTIVE_FAILURES} 次，已停止重試，請檢查 $LOG_FILE"
        exit 1
    fi

    sleep 5
done
