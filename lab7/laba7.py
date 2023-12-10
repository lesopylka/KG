import tkinter as tk

class BezierCurve:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()
        self.points = [(50, 200), (150, 100), (250, 300), (350, 200)]
        self.dragged_point = None
        self.last_x = None
        self.last_y = None
        self.draw_curve()
        self.canvas.bind("<Button-1>", self.move_point)
        self.canvas.bind("<B1-Motion>", self.drag_point)

    def draw_curve(self):
        self.canvas.delete("curve")
        self.canvas.create_line(self.points, tags="curve")

    def move_point(self, event):
        for i, (x, y) in enumerate(self.points):
            if abs(event.x - x) < 5 and abs(event.y - y) < 5:
                self.dragged_point = i
                self.last_x = event.x
                self.last_y = event.y
                break
        else:
            self.dragged_point = None

    def drag_point(self, event):
        if self.dragged_point is not None:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.last_x = event.x
            self.last_y = event.y
            x, y = self.points[self.dragged_point]
            self.points[self.dragged_point] = (x + dx, y + dy)
            self.draw_curve()

root = tk.Tk()
app = BezierCurve(root)
root.mainloop()