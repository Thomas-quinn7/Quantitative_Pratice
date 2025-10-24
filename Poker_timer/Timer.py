import tkinter as tk
from tkinter import ttk, messagebox
import json
import time

class PokerTimer:
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Tournament Timer")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a1a")
        
        # Default blind structure
        self.levels = [
            {"small_blind": 100, "big_blind": 200, "ante": 0, "duration": 20},
            {"small_blind": 200, "big_blind": 400, "ante": 0, "duration": 15},
            {"small_blind": 300, "big_blind": 600, "ante": 0, "duration": 15},
            {"small_blind": 500, "big_blind": 1000, "ante": 0, "duration": 15},
            {"small_blind": 1000, "big_blind": 2000, "ante": 0, "duration": 20},
            {"small_blind": 2000, "big_blind": 4000, "ante": 0, "duration": 20},
            {"small_blind": 3000, "big_blind": 6000, "ante": 0, "duration": 20},
            {"small_blind": 5000, "big_blind": 10000, "ante": 0, "duration": 20},
            {"small_blind": 10000, "big_blind": 20000, "ante": 0, "duration": 20},
            {"small_blind": 20000, "big_blind": 40000, "ante": 0, "duration": 20},
            {"small_blind": 40000, "big_blind": 80000, "ante": 0, "duration": 20},
            {"small_blind": 70000, "big_blind": 150000, "ante": 0, "duration": 20},
            {"small_blind": 100000, "big_blind": 200000, "ante": 0, "duration": 20},
        ]
        
        self.current_level = 0
        self.time_left = self.levels[0]["duration"] * 60
        self.is_running = False
        self.settings_window = None
        self.tick_id = None 
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        header_frame = tk.Frame(main_frame, bg="#1a1a1a")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(header_frame, text="POKER TOURNAMENT", 
                        font=("Arial", 24, "bold"), fg="#FFD700", bg="#1a1a1a")
        title.pack(side=tk.LEFT)
        
        settings_btn = tk.Button(header_frame, text="⚙ Settings", 
                                command=self.open_settings,
                                font=("Arial", 12), bg="#2d5016", fg="white",
                                relief=tk.FLAT, padx=15, pady=5, cursor="hand2")
        settings_btn.pack(side=tk.RIGHT)
        
        timer_frame = tk.Frame(main_frame, bg="#2d5016", relief=tk.RIDGE, bd=4)
        timer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.level_label = tk.Label(timer_frame, text="Level 1 of 8",
                                    font=("Arial", 16), fg="#cccccc", bg="#2d5016")
        self.level_label.pack(pady=(15, 5))
        
        self.timer_label = tk.Label(timer_frame, text="20:00",
                                    font=("Arial", 80, "bold"), fg="#FFD700", bg="#2d5016")
        self.timer_label.pack(pady=10)
        
        progress_frame = tk.Frame(timer_frame, bg="#2d5016")
        progress_frame.pack(fill=tk.X, padx=50, pady=(0, 20))
        
        self.progress_canvas = tk.Canvas(progress_frame, height=10, bg="#333333",
                                        highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X)
        self.progress_bar = self.progress_canvas.create_rectangle(
            0, 0, 0, 10, fill="#FFD700", outline=""
        )
        
        blinds_frame = tk.Frame(timer_frame, bg="#2d5016")
        blinds_frame.pack(fill=tk.X, padx=30, pady=(0, 20))
        
        sb_frame = tk.Frame(blinds_frame, bg="#1a4d1a", relief=tk.RAISED, bd=2)
        sb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(sb_frame, text="Small Blind", font=("Arial", 12),
                fg="#cccccc", bg="#1a4d1a").pack(pady=(10, 2))
        self.sb_label = tk.Label(sb_frame, text="25", font=("Arial", 28, "bold"),
                                fg="#FFD700", bg="#1a4d1a")
        self.sb_label.pack(pady=(0, 10))
        
        bb_frame = tk.Frame(blinds_frame, bg="#1a4d1a", relief=tk.RAISED, bd=3)
        bb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(bb_frame, text="Big Blind", font=("Arial", 12),
                fg="#cccccc", bg="#1a4d1a").pack(pady=(10, 2))
        self.bb_label = tk.Label(bb_frame, text="50", font=("Arial", 28, "bold"),
                                fg="#FFD700", bg="#1a4d1a")
        self.bb_label.pack(pady=(0, 10))
        
        ante_frame = tk.Frame(blinds_frame, bg="#1a4d1a", relief=tk.RAISED, bd=2)
        ante_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(ante_frame, text="Ante", font=("Arial", 12),
                fg="#cccccc", bg="#1a4d1a").pack(pady=(10, 2))
        self.ante_label = tk.Label(ante_frame, text="-", font=("Arial", 28, "bold"),
                                  fg="#FFD700", bg="#1a4d1a")
        self.ante_label.pack(pady=(0, 10))
        
        control_frame = tk.Frame(timer_frame, bg="#2d5016")
        control_frame.pack(pady=(0, 15))
        
        self.start_btn = tk.Button(control_frame, text="▶ Start",
                                   command=self.toggle_timer,
                                   font=("Arial", 14, "bold"), bg="#28a745", fg="white",
                                   relief=tk.FLAT, padx=25, pady=10, cursor="hand2")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = tk.Button(control_frame, text="↻ Reset",
                             command=self.reset_timer,
                             font=("Arial", 14, "bold"), bg="#dc3545", fg="white",
                             relief=tk.FLAT, padx=25, pady=10, cursor="hand2")
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        next_frame = tk.Frame(main_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2)
        next_frame.pack(fill=tk.X)
        
        tk.Label(next_frame, text="Next Level", font=("Arial", 14, "bold"),
                fg="#cccccc", bg="#2a2a2a").pack(pady=(10, 5))
        
        next_blinds = tk.Frame(next_frame, bg="#2a2a2a")
        next_blinds.pack(fill=tk.X, padx=30, pady=(0, 10))
        
        next_sb = tk.Frame(next_blinds, bg="#2a2a2a")
        next_sb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(next_sb, text="Small", font=("Arial", 9), fg="#888888",
                bg="#2a2a2a").pack()
        self.next_sb_label = tk.Label(next_sb, text="50", font=("Arial", 18, "bold"),
                                     fg="#cccccc", bg="#2a2a2a")
        self.next_sb_label.pack()
        
        next_bb = tk.Frame(next_blinds, bg="#2a2a2a")
        next_bb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(next_bb, text="Big", font=("Arial", 9), fg="#888888",
                bg="#2a2a2a").pack()
        self.next_bb_label = tk.Label(next_bb, text="100", font=("Arial", 18, "bold"),
                                     fg="#cccccc", bg="#2a2a2a")
        self.next_bb_label.pack()
        
        next_ante = tk.Frame(next_blinds, bg="#2a2a2a")
        next_ante.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(next_ante, text="Ante", font=("Arial", 9), fg="#888888",
                bg="#2a2a2a").pack()
        self.next_ante_label = tk.Label(next_ante, text="-", font=("Arial", 18, "bold"),
                                       fg="#cccccc", bg="#2a2a2a")
        self.next_ante_label.pack()
        
    def format_time(self, seconds):
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def format_chips(self, value):
        if value >= 1000000:
            return f"{value/1000000:.1f}M"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        else:
            return str(value)
    
    def update_display(self):
        current = self.levels[self.current_level]
        
        self.level_label.config(text=f"Level {self.current_level + 1} of {len(self.levels)}")
        
        self.timer_label.config(text=self.format_time(self.time_left))
        
        total_time = current["duration"] * 60
        elapsed = total_time - self.time_left
        progress = (elapsed / total_time) * self.progress_canvas.winfo_width()
        self.progress_canvas.coords(self.progress_bar, 0, 0, progress, 10)
        
        self.sb_label.config(text=self.format_chips(current["small_blind"]))
        self.bb_label.config(text=self.format_chips(current["big_blind"]))
        self.ante_label.config(text=self.format_chips(current["ante"]) if current["ante"] > 0 else "-")
        
        if self.current_level < len(self.levels) - 1:
            next_level = self.levels[self.current_level + 1]
            self.next_sb_label.config(text=self.format_chips(next_level["small_blind"]))
            self.next_bb_label.config(text=self.format_chips(next_level["big_blind"]))
            self.next_ante_label.config(text=self.format_chips(next_level["ante"]) 
                                       if next_level["ante"] > 0 else "-")
        else:
            self.next_sb_label.config(text="Final")
            self.next_bb_label.config(text="Level")
            self.next_ante_label.config(text="")
    
    def toggle_timer(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.start_btn.config(text="⏸ Pause")
            self.tick()
        else:
            self.start_btn.config(text="▶ Start")
            if self.tick_id:
                self.root.after_cancel(self.tick_id)
                self.tick_id = None
    
    def tick(self):
        if not self.is_running:
            self.tick_id = None
            return
            
        if self.time_left > 0:
            self.time_left -= 1
            self.update_display()
            self.tick_id = self.root.after(1000, self.tick)
        else:
            if self.current_level < len(self.levels) - 1:
                self.current_level += 1
                self.time_left = self.levels[self.current_level]["duration"] * 60
                self.root.bell() #ALARM
                self.update_display()
                self.tick_id = self.root.after(1000, self.tick)
            else:
                self.is_running = False
                self.start_btn.config(text="▶ Start")
                self.tick_id = None
                messagebox.showinfo("Tournament Complete", "All levels completed!")
    
    def reset_timer(self):
        self.is_running = False
        if self.tick_id:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None
        self.current_level = 0
        self.time_left = self.levels[0]["duration"] * 60
        self.start_btn.config(text="▶ Start")
        self.update_display()
    
    def open_settings(self):
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
            
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Blind Structure Settings")
        self.settings_window.geometry("600x500")
        self.settings_window.configure(bg="#2a2a2a")
        
        header = tk.Label(self.settings_window, text="Blind Structure",
                         font=("Arial", 18, "bold"), fg="#FFD700", bg="#2a2a2a")
        header.pack(pady=15)
        
        canvas = tk.Canvas(self.settings_window, bg="#2a2a2a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2a2a2a")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.level_entries = []
        for i, level in enumerate(self.levels):
            frame = tk.Frame(scrollable_frame, bg="#1a4d1a", relief=tk.RAISED, bd=2)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(frame, text=f"Level {i+1}", font=("Arial", 12, "bold"),
                    fg="white", bg="#1a4d1a").grid(row=0, column=0, columnspan=4, pady=5)
            
            entries = {}
            fields = [("Small Blind", "small_blind"), ("Big Blind", "big_blind"),
                     ("Ante", "ante"), ("Duration (min)", "duration")]
            
            for col, (label, key) in enumerate(fields):
                tk.Label(frame, text=label, font=("Arial", 9),
                        fg="#cccccc", bg="#1a4d1a").grid(row=1, column=col, padx=5)
                entry = tk.Entry(frame, width=10, font=("Arial", 11))
                entry.insert(0, str(level[key]))
                entry.grid(row=2, column=col, padx=5, pady=(0, 10))
                entries[key] = entry
            
            self.level_entries.append(entries)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))
        
        btn_frame = tk.Frame(self.settings_window, bg="#2a2a2a")
        btn_frame.pack(fill=tk.X, pady=10, padx=10)
        
        save_btn = tk.Button(btn_frame, text="💾 Save Changes", command=self.save_settings,
                            font=("Arial", 12, "bold"), bg="#28a745", fg="white",
                            relief=tk.FLAT, padx=20, pady=8)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(btn_frame, text="✕ Cancel", 
                              command=self.settings_window.destroy,
                              font=("Arial", 12, "bold"), bg="#6c757d", fg="white",
                              relief=tk.FLAT, padx=20, pady=8)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def save_settings(self):
        try:
            new_levels = []
            for entries in self.level_entries:
                level = {
                    "small_blind": int(entries["small_blind"].get()),
                    "big_blind": int(entries["big_blind"].get()),
                    "ante": int(entries["ante"].get()),
                    "duration": int(entries["duration"].get())
                }
                new_levels.append(level)
            
            self.levels = new_levels
            
            was_running = self.is_running
            if was_running:
                self.is_running = False
                if self.tick_id:
                    self.root.after_cancel(self.tick_id)
                    self.tick_id = None
            
            self.current_level = 0
            self.time_left = self.levels[0]["duration"] * 60
            self.start_btn.config(text="▶ Start")
            self.update_display()
            
            self.settings_window.destroy()
            messagebox.showinfo("Success", "Blind structure updated!")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields")

if __name__ == "__main__":
    root = tk.Tk()
    app = PokerTimer(root)
    root.mainloop()