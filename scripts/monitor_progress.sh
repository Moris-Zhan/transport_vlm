#!/bin/bash
# monitor_progress.sh — 監控背景執行中的正式微調訓練程序：
#   訓練程序是否還活著、CPU/記憶體使用率、磁碟剩餘空間、GPU 記憶體與使用率、訓練進度、
#   最新的 train/eval loss。
#
# 用法：
#   ./scripts/monitor_progress.sh          印一次快照就結束
#   ./scripts/monitor_progress.sh -w       每 10 秒自動刷新畫面（watch 模式，Ctrl+C 離開）
#
# 環境變數：
#   FINETUNE_LOG   訓練程序沒在跑時，改從這個日誌讀最後一次的進度/loss，預設是專案根目錄的
#                  formal_finetune_mistral_small.log（跟 run_formal_finetune.sh 的預設值一致）。
#                  訓練程序在跑時一律直接讀它的 /proc/<pid>/fd/1、fd/2（見下方說明），
#                  不管 stdout 被導向哪個檔案都看得到，不需要設定這個變數。
#
# 訓練進度條的即時性：run_formal_finetune.sh 會把每次 attempt 的輸出先寫進一個暫存檔，
# 等整個 attempt 結束才整批搬進 FINETUNE_LOG（見該腳本的說明），attempt 動輒跑好幾小時，
# 這段期間直接讀 FINETUNE_LOG 只看得到「上一次 attempt」的舊進度。這支腳本改成優先直接讀
# 訓練程序的 /proc/<pid>/fd/1 與 fd/2（不管 tqdm 進度條實際寫到 stdout 還是 stderr、也不管
# run_formal_finetune.sh 何時才把輸出搬進 FINETUNE_LOG），才能顯示「現在」真正跑到哪一步；
# 找不到活著的訓練程序時才退回顯示 FINETUNE_LOG 裡最後一次的進度條紀錄。讀 /proc/<pid>/fd/*
# 需要跟訓練程序同一個使用者（或 root）才有權限。

set -uo pipefail

# 這支腳本現在放在 scripts/ 底下，自身所在目錄不再等於專案根目錄，所以往上一層
# 才是真正的 PROJECT_DIR（Output/ checkpoint 路徑等都要相對於這裡解析）。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 這行曾被註解掉，但下面 snapshot() 找不到訓練程序時仍會讀 $LOG_FILE，搭配 set -u 會直接
# 「LOG_FILE: unbound variable」中止，整個快照印不出來。
LOG_FILE="${FINETUNE_LOG:-$PROJECT_DIR/formal_finetune_mistral_small.log}"
PROC_PATTERN="fine_tuning_vlm_trl_traffic_tune_.*\.py"

# cmdline 沒顯式帶 --model_id 時，訓練腳本會用 utils/fine_tuning/config.py 裡 TrainingConfig 的
# model_id 預設值；直接從原始碼讀出來，避免這裡寫死一份、config.py 改了預設值卻忘了同步
# （實測發生過：這裡曾寫死另一個模型的 model_id，跟 config.py 的預設值 Mistral-Small-3.2-24B 對不上）。
default_model_id() {
    sed -nE 's/^[[:space:]]*model_id: str = "([^"]+)".*/\1/p' "$PROJECT_DIR/utils/fine_tuning/config.py" 2>/dev/null | head -1
}

# 讀某個 run 資料夾存下來的 SFTConfig 欄位值（見 utils/fine_tuning/run_config_dump.py，存在
# <run>/configs/sft_config.yaml；舊版直接放在 <run>/sft_config.yaml，兩個位置都找）。
sft_config_value() {
    local run_dir="$1" key="$2" f
    for f in "$run_dir/configs/sft_config.yaml" "$run_dir/sft_config.yaml"; do
        if [ -f "$f" ]; then
            sed -nE "s/^${key}:[[:space:]]*//p" "$f" | head -1
            return
        fi
    done
}

# 從一段 tqdm 輸出裡挑出「訓練 step 進度條」那一行。HF Trainer 的訓練/評估進度條都沒有前綴
# 標籤（行首直接是百分比），但載入模型的 "Loading weights: 53%|..."、datasets 的 "Map: ..."
# 這類進度條有，先用「行首就是百分比」排除掉後者；剩下的訓練與 eval 進度條再用總 step 數區分：
# max_steps > 0 時訓練進度條的總數就是 max_steps，否則挑總數最大的那一行（eval 進度條的總數只是
# 驗證集 batch 數，遠小於整場訓練的 step 數）。
pick_train_progress_line() {
    local max_steps="$1"
    tr '\r' '\n' | grep -E '^[[:space:]]*[0-9]+%\|' | awk -v ms="$max_steps" '
        {
            if (match($0, /\| *[0-9]+\/[0-9]+/)) {
                s = substr($0, RSTART, RLENGTH); sub(/^\| */, "", s); split(s, a, "/"); total = a[2] + 0
            } else next
            if (ms + 0 > 0) { if (total == ms + 0) line = $0 }
            else if (total >= best) { best = total; line = $0 }
        }
        END { if (line != "") print line }'
}

# --- GPU 使用率彩色進度條（nvitop 風格）---
# 顏色只在真的接到終端機時輸出 ANSI escape，避免污染 log 檔或 pipe 到其他程式時混入亂碼。
if [ -t 1 ]; then
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_RED=$'\033[31m'
    C_BLUE=$'\033[34m'
    C_RESET=$'\033[0m'
else
    C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_RESET=""
fi

# 依門檻回傳顏色：<30 綠、30-70 黃、>70 紅（GPU 使用率、顯存佔用共用同一套門檻）
gpu_color() {
    local pct="$1"
    if [ "$pct" -lt 30 ]; then
        echo "$C_GREEN"
    elif [ "$pct" -lt 70 ]; then
        echo "$C_YELLOW"
    else
        echo "$C_RED"
    fi
}

# 溫度門檻：<60 綠、60-80 黃、>80 紅
temp_color() {
    local t="$1"
    if [ "$t" -lt 60 ]; then
        echo "$C_GREEN"
    elif [ "$t" -lt 80 ]; then
        echo "$C_YELLOW"
    else
        echo "$C_RED"
    fi
}

# 畫一條寬度 $2、填滿比例 $1%（0-100）的進度條，依 gpu_color 上色
draw_bar() {
    local pct="$1" width="$2" filled empty color bar
    filled=$(( pct * width / 100 ))
    # pct 是 0 以外的值時至少填滿 1 格，不然低於「1/width」(這裡是 5%) 的非零進度會無條件捨去
    # 成 0 格、整條 bar 看起來跟真的 0% 一模一樣。
    (( filled == 0 && pct > 0 )) && filled=1
    (( filled > width )) && filled=$width
    (( filled < 0 )) && filled=0
    empty=$(( width - filled ))
    color=$(gpu_color "$pct")
    printf -v bar '%*s' "$filled" ''
    printf '%s%s%s' "$color" "${bar// /█}" "$C_RESET"
    printf -v bar '%*s' "$empty" ''
    printf '%s' "${bar// /░}"
}

# 跟 draw_bar 一樣畫寬度 $2、填滿比例 $1% 的進度條，但顏色固定用 $3（不依 gpu_color 門檻變色）。
# 訓練 step 進度條用這個、固定畫藍色，跟依負載變色的 GPU 使用率/顯存 bar 意義不同，
# 所以另外寫一個函式而不是動 draw_bar。
draw_bar_fixed_color() {
    local pct="$1" width="$2" color="$3" filled empty bar
    filled=$(( pct * width / 100 ))
    # 同 draw_bar：pct>0 時至少填滿 1 格，避免低於「1/width」的非零進度整條看起來像 0%。
    (( filled == 0 && pct > 0 )) && filled=1
    (( filled > width )) && filled=$width
    (( filled < 0 )) && filled=0
    empty=$(( width - filled ))
    printf -v bar '%*s' "$filled" ''
    printf '%s%s%s' "$color" "${bar// /█}" "$C_RESET"
    printf -v bar '%*s' "$empty" ''
    printf '%s' "${bar// /░}"
}

# 抓「這一瞬間」的整體系統 CPU 使用率（不是單一程序的 %CPU）：讀兩次 /proc/stat 各欄位的
# 累計 tick 數，中間隔 0.2 秒，用差值算出這段時間裡 CPU 忙碌的比例。/proc/stat 第一行
# "cpu  user nice system idle iowait irq softirq steal guest guest_nice" 裡，
# busy = 總 tick 差 - (idle+iowait) 的 tick 差；iowait 不算忙碌（CPU 其實在等 I/O）。
cpu_usage_pct() {
    local u1 n1 s1 i1 io1 irq1 sirq1 steal1 u2 n2 s2 i2 io2 irq2 sirq2 steal2
    local total1 total2 idle1 idle2 diff_total diff_idle
    read -r _ u1 n1 s1 i1 io1 irq1 sirq1 steal1 _ < /proc/stat
    sleep 0.2
    read -r _ u2 n2 s2 i2 io2 irq2 sirq2 steal2 _ < /proc/stat
    total1=$(( u1 + n1 + s1 + i1 + io1 + irq1 + sirq1 + steal1 ))
    total2=$(( u2 + n2 + s2 + i2 + io2 + irq2 + sirq2 + steal2 ))
    idle1=$(( i1 + io1 ))
    idle2=$(( i2 + io2 ))
    diff_total=$(( total2 - total1 ))
    diff_idle=$(( idle2 - idle1 ))
    if [ "$diff_total" -le 0 ]; then
        echo 0
    else
        echo $(( 100 * (diff_total - diff_idle) / diff_total ))
    fi
}

# 從「正在跑的訓練程序」的 cmdline 反推它自己算出來的 output_dir，邏輯對應
# utils/fine_tuning/run_management.py 的 next_and_latest_run_index() 加上三支
# fine_tuning_vlm_trl_traffic_tune_*.py 共用的 run_index 選擇規則：
#   output_dir = f"Output/{model_id}-lora-run{run_index}"
#   run_index  = latest+1（有 --new_run，或這個 model_id 還沒有任何既有 run）否則沿用 latest
# 這樣才不會抓到「同一個 Output/ 底下、別的舊 run」的 checkpoint 當成目前這次訓練的進度。
# 抓不到活著的訓練程序、或 cmdline 沒帶 --model_id 時回傳空字串，呼叫端會自行退回舊行為。
current_training_output_dir() {
    local pid="$1" cmdline_raw arg model_id="" new_run=0 next_is_model_id=0
    [ -z "$pid" ] && return 1
    cmdline_raw=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null)
    [ -z "$cmdline_raw" ] && return 1
    while IFS= read -r arg; do
        if [ "$next_is_model_id" -eq 1 ]; then
            model_id="$arg"
            next_is_model_id=0
            continue
        fi
        case "$arg" in
            --model_id) next_is_model_id=1 ;;
            --model_id=*) model_id="${arg#--model_id=}" ;;
            --new_run) new_run=1 ;;
        esac
    done <<< "$cmdline_raw"
    # cmdline 沒顯式帶 --model_id 時，套用 utils/fine_tuning/config.py 裡 model_id 的預設值。
    [ -z "$model_id" ] && model_id="$(default_model_id)"
    [ -z "$model_id" ] && return 1

    local prefix="${model_id}-lora-run" latest=0 name idx
    while IFS= read -r name; do
        name="$(basename "$name")"
        [[ "$name" == "$prefix"* ]] || continue
        idx="${name#$prefix}"
        [[ "$idx" =~ ^[0-9]+$ ]] || continue
        (( idx > latest )) && latest=$idx
    done < <(ls -1d "$PROJECT_DIR"/Output/*/ 2>/dev/null)

    # 訓練腳本是在「啟動當下」決定 run_index 的；這支監控腳本是在訓練跑起來之後才重算，
    # 那時 --new_run 開的新資料夾通常已經建好、自己就是「最大 index」了，若照啟動時的規則
    # 再 +1 會指到一個不存在的 run{N+1}（實測：run3 在跑時算成 run4，checkpoint 永遠顯示無）。
    # 這裡用「最大 index 那個資料夾的修改時間是否晚於訓練程序啟動時間」判斷它是不是這次
    # 訓練自己建的：是 -> 就是它；否 -> 這次的資料夾還沒建（還在載入模型/資料），才用 latest+1。
    local run_index proc_start dir_mtime latest_dir="$PROJECT_DIR/Output/${prefix}${latest}"
    if [ "$new_run" -eq 1 ] || [ "$latest" -eq 0 ]; then
        run_index=$((latest + 1))
        if [ "$latest" -gt 0 ] && [ -d "$latest_dir" ]; then
            proc_start=$(date -d "$(ps -o lstart= -p "$pid" 2>/dev/null)" +%s 2>/dev/null)
            dir_mtime=$(stat -c %Y "$latest_dir" 2>/dev/null)
            if [ -n "$proc_start" ] && [ -n "$dir_mtime" ] && [ "$dir_mtime" -ge "$proc_start" ]; then
                run_index=$latest
            fi
        fi
    else
        run_index=$latest
    fi
    echo "$PROJECT_DIR/Output/${prefix}${run_index}"
}

snapshot() {
    echo "=================================================================="
    echo " 微調訓練監控快照 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="

    echo
    echo "--- 訓練程序 ---"
    local main_pid pid
    main_pid=""
    # pgrep -f 比對的是完整命令列字串；如果這支腳本是被其他 shell 包起來執行（例如
    # `bash -c "... python fine_tuning_vlm_trl_traffic_tune_glm.py ..."`），那個外層 bash
    # 的命令列一樣會包含腳本檔名字串、跟著一起被 pgrep 抓到，但它不是真正在跑訓練、CPU/MEM
    # 會顯示 0（實測發生過）。這裡用 /proc/PID/exe 指向的實際執行檔過濾掉這種假陽性，只留下
    # 真的是 python 直譯器的那個 PID。
    for pid in $(pgrep -f "$PROC_PATTERN"); do
        if readlink -f "/proc/$pid/exe" 2>/dev/null | grep -q '/python[0-9.]*$'; then
            main_pid="$pid"
            break
        fi
    done
    if [ -n "$main_pid" ]; then
        echo "PID    ELAPSED   %CPU  %MEM  CMD"
        ps -p "$main_pid" -o pid,etime,%cpu,%mem,cmd --no-headers
        echo "狀態：✅ 運行中"
    else
        echo "狀態：⚠️ 找不到訓練程序（可能已完成、被中斷、或尚未啟動）"
    fi

    # 先決定「目前這次訓練」的 run 資料夾：下面挑訓練進度條需要它的 max_steps，最後的
    # checkpoint 區塊也用同一個資料夾。有偵測到活著的訓練程序時，用
    # current_training_output_dir() 反推它自己的 output_dir；抓不到（程序不在跑、或還沒建好
    # 資料夾）才退回「Output/ 底下最近修改的 run 資料夾」。
    local latest_run_dir=""
    if [ -n "$main_pid" ]; then
        latest_run_dir=$(current_training_output_dir "$main_pid")
    fi
    if [ -z "$latest_run_dir" ] || [ ! -d "$latest_run_dir" ]; then
        latest_run_dir=$(ls -dt "$PROJECT_DIR"/Output/*/ 2>/dev/null | head -1)
        latest_run_dir="${latest_run_dir%/}"
    fi
    local max_steps=""
    [ -n "$latest_run_dir" ] && max_steps=$(sft_config_value "$latest_run_dir" max_steps)
    [ -z "$max_steps" ] && max_steps=-1

    # 訓練程序目前真正在寫的檔案就是它自己的 fd 1/2 指向的檔案，不管那是 stdout 還是
    # stderr（HF Trainer 的 tqdm 進度條預設寫 stderr）、也不管 run_formal_finetune.sh
    # 什麼時候才把它搬進 FINETUNE_LOG。用 tail -c 只抓檔案尾端，避免訓練跑久了這個檔案長到
    # 幾百 MB、每次快照都整份掃過；抓 256KB 是為了涵蓋「訓練進度條之後緊接著一整段 eval
    # 進度條輸出」的情況，4KB 會只剩 eval 進度條、看不到訓練進度。
    local recent_output="" fd
    if [ -n "$main_pid" ]; then
        for fd in 1 2; do
            if [ -r "/proc/$main_pid/fd/$fd" ]; then
                recent_output+=$(tail -c 262144 "/proc/$main_pid/fd/$fd" 2>/dev/null)
                recent_output+=$'\n'
            fi
        done
    elif [ -f "$LOG_FILE" ]; then
        recent_output=$(tail -c 262144 "$LOG_FILE" 2>/dev/null)
    fi

    echo
    echo "--- 訓練進度（目前這次 attempt 的即時進度條）---"
    [ -n "$latest_run_dir" ] && echo "Run：$latest_run_dir（max_steps=$max_steps）"
    local progress_line=""
    # 去掉黏在進度條後面的 loss dict（下面「最新 loss」區塊另外顯示），只留進度條本身。
    progress_line=$(pick_train_progress_line "$max_steps" <<< "$recent_output" | sed "s/{'.*//")
    if [ -n "$progress_line" ]; then
        [ -z "$main_pid" ] && echo "（訓練程序未運行，改顯示日誌 $LOG_FILE 裡最後一次的進度條紀錄）"
        echo "$progress_line"
    elif [ -n "$main_pid" ]; then
        echo "（訓練程序運行中，但暫時讀不到訓練進度條——可能剛啟動、還在載入模型/資料）"
    else
        echo "（訓練程序未運行，日誌 $LOG_FILE 裡也沒有訓練進度條紀錄）"
    fi

    # 額外畫一條藍色的訓練 step 進度條：從上面那行 tqdm 文字（例如
    # "42%|████▏     | 210/500 [00:30<00:40, 7.25it/s, loss=0.123]"）裡抓出
    # 「目前 step/總 step」（210/500）算百分比，固定用藍色跟 GPU 那組依負載變色的
    # bar 區分開來。
    if [ -n "$progress_line" ] && [[ "$progress_line" =~ ([0-9]+)/([0-9]+) ]]; then
        local step_cur="${BASH_REMATCH[1]}" step_total="${BASH_REMATCH[2]}" step_pct=0 step_bar_width=20
        if [ "$step_total" -gt 0 ] 2>/dev/null; then
            step_pct=$(( step_cur * 100 / step_total ))
            printf "Step  ["
            draw_bar_fixed_color "$step_pct" "$step_bar_width" "$C_BLUE"
            printf "] %3s%%  (%s / %s step)\n" "$step_pct" "$step_cur" "$step_total"
        fi
    fi

    # HF Trainer 每 logging_steps / eval_steps 會印一行 Python dict（例如
    # {'loss': '1.234', 'grad_norm': ..., 'learning_rate': ..., 'epoch': '0.01'}），各取最後一行。
    local last_train_log last_eval_log
    # 這行 dict 常常緊接在 tqdm 進度條文字後面、沒有換行（實測："50%|█████| 5/10 [...]{'loss': ...}"），
    # 所以用 grep -o 從行中間抓出整段 dict，而不是要求它在行首。
    last_train_log=$(tr '\r' '\n' <<< "$recent_output" | grep -oE "\{'loss'[^}]*\}" | tail -1)
    last_eval_log=$(tr '\r' '\n' <<< "$recent_output" | grep -oE "\{'eval_loss'[^}]*\}" | tail -1)
    echo
    echo "--- 最新 loss ---"
    echo "train：${last_train_log:-（尚無紀錄）}"
    echo "eval ：${last_eval_log:-（尚無紀錄）}"

    # if [ -f "$LOG_FILE" ]; then
    #     local retry_lines
    #     retry_lines=$(tr '\r' '\n' < "$LOG_FILE" | grep -E "^===|Traceback|Error|OOM|out of memory" | tail -5)
    #     if [ -n "$retry_lines" ]; then
    #         echo
    #         echo "--- 最近的重試/錯誤紀錄 ---"
    #         echo "$retry_lines"
    #     fi
    # else
    #     echo "找不到日誌檔：$LOG_FILE"
    #     echo "（設定 FINETUNE_LOG 環境變數指到實際的日誌路徑）"
    # fi

    echo
    echo "--- CPU / 系統記憶體 ---"
    local sys_bar_width=20 cpu_pct
    cpu_pct=$(cpu_usage_pct)
    printf "CPU  ["
    draw_bar "$cpu_pct" "$sys_bar_width"
    printf "] %3s%%\n" "$cpu_pct"

    local mem_total mem_used mem_pct mem_total_gib mem_used_gib
    read -r mem_total mem_used < <(free -b | awk '/^Mem:/{print $2, $3}')
    mem_pct=0
    if [ "${mem_total:-0}" -gt 0 ] 2>/dev/null; then
        mem_pct=$(( mem_used * 100 / mem_total ))
    fi
    mem_total_gib=$(awk -v b="${mem_total:-0}" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
    mem_used_gib=$(awk -v b="${mem_used:-0}" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
    printf "MEM  ["
    draw_bar "$mem_pct" "$sys_bar_width"
    printf "] %3s%%  (%sGi / %sGi)\n" "$mem_pct" "$mem_used_gib" "$mem_total_gib"

    echo
    echo "--- 磁碟空間（共用儲存）---"
    local disk_target disk_line disk_size disk_used disk_pct disk_mount
    disk_target="/mnt/moris"
    df -h "$disk_target" >/dev/null 2>&1 || disk_target="$PROJECT_DIR"
    disk_line=$(df -h "$disk_target" 2>/dev/null | tail -1)
    disk_size=$(awk '{print $2}' <<< "$disk_line")
    disk_used=$(awk '{print $3}' <<< "$disk_line")
    disk_pct=$(awk '{print $5}' <<< "$disk_line" | tr -d '%')
    disk_mount=$(awk '{print $6}' <<< "$disk_line")
    printf "DISK ["
    draw_bar "${disk_pct:-0}" "$sys_bar_width"
    printf "] %3s%%  (%s / %s)  掛載點：%s\n" "${disk_pct:-0}" "$disk_used" "$disk_size" "$disk_mount"

    echo
    echo "--- GPU ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_csv bar_width=20
        # nounits 拿掉 "MiB"/"%" 等單位方便直接做數字運算；用 sed 把 ", " 換成 "|"
        # 而不是直接用逗號當分隔字元，這樣 GPU 名稱裡的空白（如 "NVIDIA GeForce RTX 4090"）
        # 才不會被拆成好幾個欄位。
        gpu_csv=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | sed -E 's/, */|/g')
        if [ -n "$gpu_csv" ]; then
            local idx name util mem_used mem_total temp mem_pct
            while IFS='|' read -r idx name util mem_used mem_total temp; do
                [ -z "$idx" ] && continue
                mem_pct=0
                if [ "$mem_total" -gt 0 ] 2>/dev/null; then
                    mem_pct=$(( mem_used * 100 / mem_total ))
                fi
                printf "GPU%s %-28s [" "$idx" "$name"
                draw_bar "$util" "$bar_width"
                printf "] %3s%%  MEM[" "$util"
                draw_bar "$mem_pct" "$bar_width"
                printf "] %6sMiB / %6sMiB  %s%s°C%s\n" \
                    "$mem_used" "$mem_total" "$(temp_color "$temp")" "$temp" "$C_RESET"
            done <<< "$gpu_csv"
        else
            echo "讀不到 GPU 資訊（nvidia-smi 沒有輸出）"
        fi
    else
        echo "找不到 nvidia-smi"
    fi

    echo
    echo "--- 最新 checkpoint ---"
    local latest_ckpt save_steps_hint=""
    # 只抓「目前正在訓練」那個 run 資料夾（latest_run_dir，snapshot() 開頭已決定）裡的
    # checkpoint，不要誤抓 Output/ 底下其他舊 run 留下來的 checkpoint。
    latest_ckpt=""
    if [ -n "$latest_run_dir" ]; then
        latest_ckpt=$(ls -dt "$latest_run_dir"/checkpoint-* 2>/dev/null | head -1)
    fi
    if [ -n "$latest_ckpt" ]; then
        echo "$latest_ckpt"
    else
        # save_steps 現在每個 run 可各自設定（見 utils/fine_tuning/config.py 的
        # --save_steps），不再是寫死的 HF 預設值 500；優先讀這個 run 自己存的
        # sft_config.yaml（見 utils/fine_tuning/run_config_dump.py）來顯示實際生效的值。
        [ -n "$latest_run_dir" ] && save_steps_hint=$(sft_config_value "$latest_run_dir" save_steps)
        if [ -n "$latest_run_dir" ]; then
            echo "無（${latest_run_dir} 這個 run 尚未存到第一個 checkpoint${save_steps_hint:+，每 ${save_steps_hint} step 存一次}）"
        else
            echo "無"
        fi
    fi

    echo "=================================================================="
}

if [ "${1:-}" = "-w" ] || [ "${1:-}" = "--watch" ]; then
    while true; do
        clear
        snapshot
        sleep 10
    done
else
    snapshot
fi
