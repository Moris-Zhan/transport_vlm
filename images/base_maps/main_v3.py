import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
import os

class IntersectionMarker:
    def __init__(self, root):
        self.root = root
        self.root.title("十字路口位置標記程式")
        self.root.geometry("1200x800")
        
        # 初始化變數
        self.image = None
        self.photo = None
        self.canvas = None
        self.current_image_path = ""
        self.scale_factor = 1.0
        
        # 定義所有路口位置類型（按照您提供的順序）
        self.position_types = [
            "路口中端", "路段-西-往東", "鄰近路口-西", "路口近端-西", "路口遠端-東",
            "路段-東-往東", "路段-東-往西", "鄰近路口-東", "路口近端-東", "路口遠端-西",
            "路段-西-往西", "路段-南-往北", "鄰近路口-南", "路口近端-南", "路口遠端-北",
            "路段-北-往北", "路段-北-往南", "鄰近路口-北", "路口近端-北", "路口遠端-南",
            "路段-南-往南"
        ]
        
        # 儲存標記點的座標
        self.coordinates = {}
        self.current_position_index = 0
        self.marked_points = []  # 儲存在畫布上繪製的點
        
        # 拖動相關變數
        self.dragging_point = None
        self.drag_data = {"x": 0, "y": 0}
        self.drag_point_index = -1
        
        # CSV資料儲存
        self.csv_data = []
        
        self.setup_ui()

        # self.img_path = "C:/Project/Git/transport_vlm/images/base_maps(v2)/正交十字路口-中央分隔雙向2車道.jpg"
        # self.load_image()

        self.csv_path = "C:/Project/Git/transport_vlm/images/base_maps(v2)/底圖info.csv"
        self.load_csv()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 檔案選擇按鈕
        ttk.Button(control_frame, text="選擇圖片", command=self.load_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # 當前標記位置顯示
        self.position_label = ttk.Label(control_frame, text="請先選擇圖片", font=("Arial", 12, "bold"))
        self.position_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # 右側控制按鈕
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="重設當前圖片", command=self.reset_current_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="清除標記", command=self.clear_marks).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="下一張圖片", command=self.next_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="儲存CSV", command=self.save_csv).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="載入CSV", command=self.load_csv).pack(side=tk.LEFT)
        
        # 圖片顯示區域
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 創建畫布和滾動條
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # 佈局滾動條和畫布
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 綁定滑鼠點擊事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # 綁定滑鼠移動事件
        self.canvas.bind("<Motion>", self.on_mouse_motion)
        # 綁定拖動相關事件
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # 進度顯示
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_label = ttk.Label(progress_frame, text="進度: 0/21")
        self.progress_label.pack(side=tk.LEFT)
        
        # 座標顯示（左下角）
        self.coordinate_label = ttk.Label(progress_frame, text="座標: (0, 0)", font=("Arial", 10))
        self.coordinate_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=21)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
    def load_image(self):
        """載入圖片"""
        file_path = filedialog.askopenfilename(
            title="選擇十字路口圖片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        # file_path = self.img_path

        # print(file_path)
        
        if file_path:
            self.current_image_path = file_path
            self.reset_current_image()
            self.display_image()
            
    def display_image(self):
        """顯示圖片到畫布上 (固定 1000x1000)"""
        if not self.current_image_path:
            return

        # 讀取圖片
        image = cv2.imdecode(np.fromfile(file=self.current_image_path, dtype=np.uint8),
                            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🚩強制縮放成 1000x1000
        fixed_size = (1000, 1000)
        self.image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)

        # 轉換為 PIL 格式再顯示
        pil_image = Image.fromarray(self.image)
        self.photo = ImageTk.PhotoImage(pil_image)

        # 🚩scrollregion 也固定 1000x1000
        self.canvas.config(scrollregion=(0, 0, fixed_size[0], fixed_size[1]))

        # 清空畫布並繪製圖片
        self.canvas.delete("all")
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # 嘗試從 CSV 載入座標
        self.load_coordinates_from_csv()

        # 重新繪製標記點
        self.redraw_points()

        # 更新當前位置顯示
        self.update_position_label()

    def on_canvas_click(self, event):
        """處理畫布點擊事件"""
        if not self.current_image_path or self.current_position_index >= len(self.position_types):
            return
            
        # 獲取點擊座標（考慮滾動位置）
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 轉換為原始圖片座標
        original_x = int(x / self.scale_factor)
        original_y = int(y / self.scale_factor)
        
        # 儲存座標
        position_name = self.position_types[self.current_position_index]
        self.coordinates[position_name] = (original_x, original_y)
        
        # 在畫布上繪製標記點
        point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, 
                                         fill="red", outline="darkred", width=2,
                                         tags="draggable")
        # 顯示標記編號
        text_id = self.canvas.create_text(x, y-15, text=str(self.current_position_index + 1), 
                                        fill="red", font=("Arial", 10, "bold"),
                                        tags="draggable")
        # 顯示座標
        coord_text_id = self.canvas.create_text(x, y+15, text=f"({original_x},{original_y})", 
                                               fill="blue", font=("Arial", 8),
                                               tags="draggable")
        
        self.marked_points.append((point_id, text_id, coord_text_id))
        
        # 移動到下一個位置
        self.current_position_index += 1
        self.update_position_label()
        self.update_progress()
        
        # 如果所有位置都標記完成
        if self.current_position_index >= len(self.position_types):
            messagebox.showinfo("完成", "所有位置已標記完成！")

        # print("click: ", f"座標: ({x}, {y})")
        # print("click: ", self.marked_points)
        # print("scale_factor: " , self.scale_factor)

    def clear_marks(self):
        """清除當前圖片的所有標記點"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "請先載入圖片")
            return
            
        if not self.coordinates and self.current_position_index == 0:
            messagebox.showinfo("提示", "當前圖片沒有標記點")
            return
            
        # 詢問使用者確認
        # result = messagebox.askyesno("確認清除", "確定要清除當前圖片的所有標記點嗎？\n(這將同時清除CSV中對應的記錄)")
        result = True
        if result:
            # 清除座標資料
            self.coordinates.clear()
            self.current_position_index = 0
            
            # 清除畫布上的標記點
            if self.canvas:                
                # 移除所有標記元素
                for elements in self.marked_points:
                    for element_id in elements:
                        self.canvas.delete(element_id)
                        
            self.marked_points.clear()
            
            # 從CSV資料中移除對應圖片的記錄
            # self.remove_image_from_csv()
            
            # 更新介面
            self.update_position_label()
            self.update_progress()
            
            # messagebox.showinfo("完成", "已清除所有標記點和CSV記錄")
    

    def on_mouse_motion(self, event):
        """處理滑鼠移動事件，更新座標顯示"""
        if not self.current_image_path:
            self.coordinate_label.config(text="座標: (0, 0)")
            return
            
        # 獲取滑鼠座標（考慮滾動位置）
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 轉換為原始圖片座標
        original_x = int(x / self.scale_factor)
        original_y = int(y / self.scale_factor)
        
        # 更新座標顯示
        self.coordinate_label.config(text=f"座標: ({original_x}, {original_y})")
        
    def on_button_press(self, event):
        """處理滑鼠按下事件"""
        if not self.current_image_path:
            return
            
        # 獲取點擊座標
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 檢查是否點擊在現有的標記點上
        clicked_item = self.canvas.find_closest(x, y)[0]
        
        # 檢查點擊的物件是否有 draggable 標籤
        if "draggable" in self.canvas.gettags(clicked_item):
            # 找到對應的標記點組
            for i, (point_id, text_id, coord_text_id) in enumerate(self.marked_points):
                if clicked_item in (point_id, text_id, coord_text_id):
                    self.dragging_point = (point_id, text_id, coord_text_id)
                    self.drag_point_index = i
                    self.drag_data["x"] = x
                    self.drag_data["y"] = y
                    
                    # 改變拖動中的顏色
                    self.canvas.itemconfig(point_id, fill="orange", outline="darkorange")
                    return
        
        # 如果沒有點擊到標記點，執行正常的標記操作
        self.on_canvas_click(event)
        
    def on_drag_motion(self, event):
        """處理拖動過程中的滑鼠移動"""
        if not self.dragging_point:
            return
            
        # 獲取當前滑鼠位置
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 計算移動距離
        dx = x - self.drag_data["x"]
        dy = y - self.drag_data["y"]
        
        # 移動標記點組的所有元素
        point_id, text_id, coord_text_id = self.dragging_point
        self.canvas.move(point_id, dx, dy)
        self.canvas.move(text_id, dx, dy)
        self.canvas.move(coord_text_id, dx, dy)
        
        # 更新座標文字
        original_x = int(x / self.scale_factor)
        original_y = int(y / self.scale_factor)
        self.canvas.itemconfig(coord_text_id, text=f"({original_x},{original_y})")
        
        # 更新拖動資料
        self.drag_data["x"] = x
        self.drag_data["y"] = y

        self.coordinate_label.config(text=f"座標: ({int(x)}, {int(y)})")
        
    def on_button_release(self, event):
        """處理滑鼠釋放事件"""
        if not self.dragging_point:
            return
            
        # 獲取最終位置
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 轉換為原始圖片座標
        original_x = int(x / self.scale_factor)
        original_y = int(y / self.scale_factor)
        
        # 更新對應位置的座標資料
        if 0 <= self.drag_point_index < len(self.position_types):
            position_name = self.position_types[self.drag_point_index]
            self.coordinates[position_name] = (original_x, original_y)
        
        # 恢復原始顏色
        point_id, text_id, coord_text_id = self.dragging_point
        self.canvas.itemconfig(point_id, fill="red", outline="darkred")
        
        # 清除拖動狀態
        self.dragging_point = None
        self.drag_point_index = -1
            
    def redraw_points(self):
        """重新繪製標記點"""
        self.marked_points.clear()
        
        for i, position_name in enumerate(self.position_types[:self.current_position_index]):
            if position_name in self.coordinates:
                x, y = self.coordinates[position_name]  # 這裡不要乘 scale 了

                point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5,
                                                   fill="red", outline="darkred", width=2)
                text_id = self.canvas.create_text(x, y-15, text=str(i + 1),
                                                  fill="red", font=("Arial", 10, "bold"))
                coord_text_id = self.canvas.create_text(x, y+15, text=f"({x},{y})",
                                                        fill="blue", font=("Arial", 8))
                self.marked_points.append((point_id, text_id, coord_text_id))
    
    def update_position_label(self):
        """更新當前位置標籤"""
        if self.current_position_index < len(self.position_types):
            current_position = self.position_types[self.current_position_index]
            self.position_label.config(text=f"請點選: {current_position} ({self.current_position_index + 1}/21)")
        else:
            self.position_label.config(text="所有位置已標記完成")
            
    def update_progress(self):
        """更新進度條"""
        self.progress_bar['value'] = self.current_position_index
        self.progress_label.config(text=f"進度: {self.current_position_index}/21")
        
    def reset_current_image(self):
        """重設當前圖片的標記"""
        self.coordinates.clear()
        self.current_position_index = 0
        self.marked_points.clear()
        
        if self.canvas:
            self.canvas.delete("all")
            
        if self.current_image_path:
            self.display_image()
            
        self.update_position_label()
        self.update_progress()
        
    def next_image(self):
        """處理下一張圖片"""
        if self.current_position_index < len(self.position_types):
            result = messagebox.askyesno("確認", "當前圖片尚未完成標記，是否要儲存並繼續下一張？")
            if not result:
                return
                
        # 儲存當前圖片的資料
        if self.current_image_path and self.coordinates:
            self.save_current_image_data()
            
        # 選擇新圖片
        self.load_image()
        
    def save_current_image_data(self):
        """儲存當前圖片的資料到內部儲存"""
        if not self.current_image_path:
            return
        
        # 檢查是否已存在相同圖片的資料
        image_name = os.path.basename(self.current_image_path)
        existing_index = -1
        
        for i, row in enumerate(self.csv_data):
            if row and os.path.basename(row[0]) == image_name:
                existing_index = i
                break
                
        # 準備CSV行資料
        # row_data = [self.current_image_path]
        row_data = [ "images/base_maps/" + os.path.basename(self.current_image_path) ]
        
        # 按照位置順序添加座標
        for position_name in self.position_types:
            if position_name in self.coordinates:
                coord = self.coordinates[position_name]
                row_data.append(f"[({coord[0]},{coord[1]})]")
            else:
                row_data.append("")  # 未標記的位置留空
        
        # 如果找到相同圖片，更新資料；否則新增
        if existing_index >= 0:
            self.csv_data[existing_index] = row_data
            messagebox.showinfo("更新", f"已更新 {image_name} 的標記資料")
        else:
            self.csv_data.append(row_data)
            messagebox.showinfo("儲存", f"已儲存 {image_name} 的標記資料")
        
    def save_csv(self):
        """儲存CSV檔案"""
        # 如果當前有圖片且有標記，先儲存
        if self.current_image_path and self.coordinates:
            self.save_current_image_data()
            
        if not self.csv_data:
            messagebox.showwarning("警告", "沒有資料可以儲存")
            return
            
        # 選擇儲存位置
        file_path = filedialog.asksaveasfilename(
            title="儲存CSV檔案",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # 寫入標題行
                    header = ["Image"] + self.position_types
                    writer.writerow(header)
                    
                    # 寫入資料行
                    for row in self.csv_data:
                        writer.writerow(row)
                        
                messagebox.showinfo("成功", f"CSV檔案已儲存至: {file_path}")
                
            except Exception as e:
                messagebox.showerror("錯誤", f"儲存CSV檔案時發生錯誤: {str(e)}")
                
    def load_csv(self):
        """載入現有的CSV檔案"""
        # file_path = filedialog.askopenfilename(
        #     title="載入CSV檔案",
        #     filetypes=[("CSV files", "*.csv")]
        # )    

        file_path = self.csv_path   
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)  # 讀取標題行
                    
                    self.csv_data = []
                    for row in reader:
                        self.csv_data.append(row)
                        
                # messagebox.showinfo("成功", f"已載入CSV檔案，共 {len(self.csv_data)} 筆資料")
                
                if self.canvas:
                    self.canvas.delete("all")
                    
                if self.current_image_path:
                    self.display_image()            
                
            except Exception as e:
                messagebox.showerror("錯誤", f"載入CSV檔案時發生錯誤: {str(e)}")

    def load_coordinates_from_csv(self):
        """從CSV資料中載入當前圖片的座標"""
        if not self.current_image_path or not self.csv_data:
            return
            
        image_name = os.path.basename(self.current_image_path)
        
        # 在CSV資料中尋找當前圖片
        for row in self.csv_data:
            if row and os.path.basename(row[0]) == image_name:
                # 找到對應圖片，載入座標
                self.coordinates.clear()
                self.current_position_index = 0
                
                for i, position_name in enumerate(self.position_types):
                    # CSV中的座標欄位從索引1開始（索引0是圖片路徑）
                    if i + 1 < len(row) and row[i + 1]:
                        coord_str = row[i + 1]
                        # 解析座標字串 "[(<x>,<y>)]"
                        try:
                            # 移除方括號和括號，分割座標
                            coord_str = coord_str.strip("[]() ")
                            if ',' in coord_str:
                                x, y = coord_str.split(',')
                                x = int(x.strip()) 
                                y = int(y.strip()) 
                                # x = int(x.strip()) * self.scale_factor
                                # y = int(y.strip()) * self.scale_factor
                                self.coordinates[position_name] = (x, y)
                                self.current_position_index += 1
                        except (ValueError, IndexError) as e:
                            print(f"解析座標錯誤: {coord_str}, 錯誤: {e}")
                            continue            
                    
                self.update_position_label()
                self.update_progress()
                
                # messagebox.showinfo("載入成功", f"已載入 {image_name} 的 {len(self.coordinates)} 個標記點")
                break

def main():
    """主程式"""
    root = tk.Tk()
    app = IntersectionMarker(root)
    
    # # 顯示使用說明
    # messagebox.showinfo(
    #     "使用說明", 
    #     "1. 點選「選擇圖片」載入十字路口底圖\n" +
    #     "2. 依序點選圖片上的各個路口位置\n" +
    #     "3. 標記完成後點選「下一張圖片」或「儲存CSV」\n" +
    #     "4. 「重設當前圖片」可清除當前圖片的所有標記\n" +
    #     "5. 「載入CSV」可載入之前儲存的標記資料"
    # )
    
    root.mainloop()

if __name__ == "__main__":
    main()