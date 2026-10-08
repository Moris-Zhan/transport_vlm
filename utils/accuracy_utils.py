"""Vision-LLM 推論結果的準確率計算工具。

從 vlm_infer_test.ipynb 搬移而來:比對模型輸出文字與標註文字的字元正確率,
以及比對事件段落數(icons)、OCR 受傷人數標籤的準確率/precision/recall/F1。
"""

import re
import difflib
from collections import Counter


def extract_numbers(text):
    """從文字中提取所有數字"""
    return re.findall(r'\d+', text)


def preprocess_text(text):
    """移除句首序號並清理空白"""
    lines = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # 移除句首如 "1."、"2." 等序號
        line = re.sub(r'^\d+\.', '', line).strip()
        lines.append(line)
    # 根據句子內容排序(字典序)
    return sorted(lines)


def calculate_char_accuracy(text1, text2):
    """計算兩段文字的字元正確率(使用編輯距離)"""
    # 使用 SequenceMatcher 計算相似度
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def compare_texts_by_characters(text1, text2):
    """比對兩段文字並計算字元正確率"""
    sorted1 = preprocess_text(text1)
    sorted2 = preprocess_text(text2)

    # 貪婪最佳配對:依序為 sorted1 的每一句,在 sorted2 剩餘句子中找相似度最高的
    # 一句配對後取出。單純依字典序排序後逐一對位,只要對應句子內容有差異就可能
    # 造成排序名次錯開,進而讓後續所有句子連鎖錯位比對到不相關的句子;改用最佳
    # 配對可以避免這個問題。
    remaining2 = list(sorted2)
    pairs = []
    for t1 in sorted1:
        if remaining2:
            best_idx = max(
                range(len(remaining2)),
                key=lambda i: difflib.SequenceMatcher(None, t1, remaining2[i]).ratio(),
            )
            t2 = remaining2.pop(best_idx)
        else:
            t2 = ""
        pairs.append((t1, t2))
    # sorted2 中沒被配對到的句子,視為多預測出來、標註中不存在的句子
    for t2 in remaining2:
        pairs.append(("", t2))

    # 統計變數
    total_chars = 0
    correct_chars = 0
    correct_sent = 0

    report_details = []
    report_lines = []

    for t1, t2 in pairs:
        # 計算此句子的字元比對
        matcher = difflib.SequenceMatcher(None, t1, t2)
        matching_blocks = matcher.get_matching_blocks()

        # 計算匹配的字元數(排除最後的dummy block)
        matched = sum(block.size for block in matching_blocks[:-1])
        total = max(len(t1), len(t2))

        total_chars += total
        correct_chars += matched

        same = t1 == t2
        accuracy = (matched / total * 100) if total > 0 else 100.0

        # if same:
        #     report_lines.append("🟢 比對結果相同")
        #     report_lines.append(f"句子:{t1}")
        # else:
        if not same:
            report_details.append("🔴 比對結果不同")
            report_details.append(f"原始句子:{t1}")
            report_details.append(f"比對句子:{t2}")
            report_details.append(f"句子正確率:{accuracy:.2f}%")

            # 顯示差異細節
            diff = list(difflib.ndiff(t1, t2))
            diff_display = []
            for d in diff:
                if d.startswith('  '):  # 相同
                    diff_display.append(d[2])
                elif d.startswith('- '):  # 原文有,比對文沒有
                    diff_display.append(f"[➖{d[2]}]")
                elif d.startswith('+ '):  # 比對文有,原文沒有
                    diff_display.append(f"[➕{d[2]}]")
            if diff_display:
                report_details.append(f"差異標記:{''.join(diff_display)}")
            # report_details.append("-" * 50)
        else:
            report_details.append("🔴 比對結果相同")
            correct_sent = correct_sent + 1
        report_details.append("-" * 50)

    # 計算整體字元正確率
    overall_accuracy = (correct_chars / total_chars * 100) if total_chars > 0 else 100.0
    overall_sent_accuracy = (correct_sent / len(sorted1) * 100) if correct_sent > 0 else 0.0

    # 加入統計資訊
    report_lines.append(f"\n📊 統計資訊:")
    report_lines.append(f"總字元數:{total_chars}")
    report_lines.append(f"正確字元數:{correct_chars}")
    report_lines.append(f"錯誤字元數:{total_chars - correct_chars}")
    report_lines.append(f"\n✅ 整體字元正確率:{overall_accuracy:.2f}%")
    report_lines.append(f"\n✅ 整體句子正確率:{overall_sent_accuracy:.2f}%")
    report_lines.append("-" * 50)

    full_report = "\n".join(report_lines) + "\n".join(report_details)

    return full_report, overall_accuracy


def extract_pointer(text):
    # 提取每一段的起始數字（事件段落數）
    icons = len(re.findall(r'^\d+\.', text, re.MULTILINE))

    # 提取所有「受傷人數」後的數字
    ocr_labels = [int(x) for x in re.findall(r'受傷人數(\d+)人', text)]

    # 含有「受傷人數」的句子數量
    ocr_nums = len(ocr_labels)

    # 結果輸出
    result = (icons, ocr_nums, ocr_labels)
    return result


def compare_extraction(label: dict, pred: dict, legend_source: str):
    # icons accuracy:誤差相對於標註值的比例,夾在 [0, 1] 之間(避免預測誤差大於
    # 標註值時得到負的「準確率」),分母以 1 為下限(避免 label 為 0 時除以零)。
    icons_lbl = label['icons_lbl']
    icons_diff = abs(icons_lbl - pred['icons'])
    icons_accuracy = max(0.0, 1 - icons_diff / max(icons_lbl, 1))

    # OCR multiset compare
    true_counter = Counter(label['ocr_labels_lbl'])
    pred_counter = Counter(pred['ocr_labels'])

    tp = sum(min(true_counter[k], pred_counter.get(k, 0)) for k in true_counter)
    total_true = sum(true_counter.values())
    total_pred = sum(pred_counter.values())

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_true if total_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # FP / FN details
    # FP: 預測中多出來且未被 label 匹配的元素（包含次數）
    fp_counter = pred_counter.copy()
    for k in true_counter:
        matched = min(true_counter[k], pred_counter.get(k, 0))
        if matched > 0:
            fp_counter[k] -= matched
    # 移除 0 或負值
    fp_counter = +fp_counter  # Counter 的 unary + 會去除非正項

    # FN: label 中未被匹配到的元素（包含次數）
    fn_counter = true_counter.copy()
    for k in pred_counter:
        matched = min(true_counter.get(k, 0), pred_counter[k])
        if matched > 0:
            fn_counter[k] -= matched
    fn_counter = +fn_counter

    tp_detail = {k: min(true_counter[k], pred_counter.get(k, 0)) for k in true_counter if min(true_counter[k], pred_counter.get(k, 0)) > 0}
    fp_detail = dict(fp_counter)
    fn_detail = dict(fn_counter)

    fp = sum(fp_detail.values())
    fn = sum(fn_detail.values())

    if tp == fp == fn == 0:
        ocr_accuracy = 1.0
    else:
        ocr_accuracy = tp / (tp + fp + fn)

    result = {
        'icons_accuracy': icons_accuracy * 100,
        'ocr_tp': tp,
        'ocr_total_pred': total_pred,
        'ocr_total_true': total_true,
        'ocr_precision': precision * 100,
        'ocr_recall': recall * 100,
        'f1': f1 * 100,
        'ocr_accuracy': ocr_accuracy * 100,
        'tp_detail': tp_detail,
        'fp_detail': fp_detail,
        'fn_detail': fn_detail,
        'ocr_fp': fp,
        'ocr_fn': fn
    }

    # OCR_Report:結構化摘要 —— icons 段落 + OCR 段落(✅/❌ 標示 TP/FP/FN)+
    # precision/recall/F1/IoU 數字,取代直接把 dict repr 塞進報表的呈現方式。
    
    if legend_source == "阡陌":
        ocr_report = (
            f"【圖示數量 Icons】標註:{icons_lbl}　預測:{pred['icons']}　"
            f"誤差:{icons_diff}　命中率:{result['icons_accuracy']:.2f}%\n"
            f"\n"
            f"【OCR 受傷人數】標註:{label['ocr_labels_lbl']}　辨識:{pred['ocr_labels']}\n"
            f"✅ 命中(TP):{tp_detail}\n"
            f"❌ 多辨識(FP):{fp_detail}\n"
            f"❌ 漏辨識(FN):{fn_detail}\n"
            f"Precision:{result['ocr_precision']:.2f}%　"
            f"Recall:{result['ocr_recall']:.2f}%　"
            f"F1:{result['f1']:.2f}%　"
            f"IoU:{result['ocr_accuracy']:.2f}%"
        )
    elif legend_source == "運研所":
        ocr_report = (
            f"【圖示數量 Icons】標註:{icons_lbl}　預測:{pred['icons']}　"
            f"誤差:{icons_diff}　命中率:{result['icons_accuracy']:.2f}%\n"
            f"\n"            
        )

    return result, ocr_report
