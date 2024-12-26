import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.interpolate import UnivariateSpline
import copy
import sys

class BrightnessCurveEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Brightness Curve Editor")
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initial control points
        self.points = np.array([
            [   0.0000,   0],
            [   1.2689,   1],
            [   1.7738,   4],
            [   2.4651,   7],
            [   3.4113,  12],
            [   4.7065,  17],
            [   6.4795,  24],
            [   8.9066,  31],
            [  12.2289,  39],
            [  16.7768,  49],
            [  23.0023,  59],
            [  31.5244,  70],
            [  43.1900,  82],
            [  59.1588,  95],
            [  81.0182, 109],
            [ 110.9412, 124],
            [ 151.9022, 140],
            [ 207.9728, 156],
            [ 284.7269, 174],
            [ 389.7940, 193],
            [ 533.6184, 212],
            [ 730.4969, 233],
            [1000.0000, 255]
        ])
                
        # History logging
        self.history = [copy.deepcopy(self.points)]
        self.current_history_idx = 0
        self.max_history = 50  # Maximum number of history records
        
        # Create main frame
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib chart
        self.fig = plt.Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()  # Create secondary y-axis and save it
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Automatically adjusts the chart size
        self.fig.set_tight_layout(True)
        
        # Create bottom frame
        self.bottom_frame = ttk.Frame(self.frame)
        self.bottom_frame.pack(fill=tk.X, pady=5)
        
        # Left side instructions text
        ttk.Label(self.bottom_frame, 
                text="Left Click : Add/Move Point ;   "
                    "Delete : Delete Selected Point ;   "
                    "F : Auto Fit ;   "
                    "Ctrl+Z : Undo ;   "
                    "Ctrl+Y : Redo ."
                ).pack(side=tk.LEFT, padx=5)
        
        # Right side info label and button
        self.info_label = ttk.Label(self.bottom_frame, text="")
        self.info_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(self.bottom_frame, text="Export Parameters", 
                  command=self.export_parameters).pack(side=tk.RIGHT, padx=5)
        
        # Initialize variables
        self.selected_point = None
        self.dragging = False
        
        # Update chart
        self.update_plot()
        
        # Bind events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.root.bind('<Delete>', self.on_delete)
        self.root.bind('f', self.auto_fit)
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Control-y>', self.redo)
        
        # Bind window resize event
        self.frame.bind('<Configure>', self.on_resize)

    def transform_x_to_log(self, x):
        """Convert x values to logarithmic space, supports array input."""
        result = np.zeros_like(x, dtype=float)
        result[x <= 0] = -0.1  # Assign -0.1 for non-positive values
        result[x > 0] = np.log10(x[x > 0])
        return result
    
    def transform_x_from_log(self, log_x):
        """Convert from logarithmic space back to actual x values, supports array input."""
        result = np.zeros_like(log_x, dtype=float)
        result[log_x < 0] = 0  # Assign 0 for negative logarithmic values
        result[log_x >= 0] = 10 ** log_x[log_x >= 0]
        return result

    def add_to_history(self):
        """Add current state to history logging."""
        self.history = self.history[:self.current_history_idx + 1]
        self.history.append(copy.deepcopy(self.points))
        if len(self.history) > self.max_history:
            self.history.pop(0)  # Remove oldest history if exceeds limit
        else:
            self.current_history_idx += 1
    
    def undo(self, event=None):
        """Perform undo operation."""
        if self.current_history_idx > 0:
            self.current_history_idx -= 1
            self.points = copy.deepcopy(self.history[self.current_history_idx])  # Restore previous points
            self.update_plot()
    
    def redo(self, event=None):
        """Perform redo operation."""
        if self.current_history_idx < len(self.history) - 1:
            self.current_history_idx += 1
            self.points = copy.deepcopy(self.history[self.current_history_idx])  # Restore next points in history
            self.update_plot()
    
    def update_plot(self):
        """Update the plot with current points and fitted curves."""
        self.ax.clear()
        self.ax2.clear()  # Clear secondary axis

        # Set y-axis limits to 0-255
        self.ax.set_ylim(0, 255)
        self.ax2.set_ylim(0, 255)

        # Prepare plotting data
        plot_points = self.points.copy()

        # Convert x values to logarithmic space
        transformed_x = np.zeros_like(plot_points[:, 0])
        mask = plot_points[:, 0] > 0
        transformed_x[mask] = np.log10(plot_points[mask, 0])

        # Plot main curve
        self.ax.plot(transformed_x, plot_points[:, 1], 'b-')

        # Plot fitted suggestion curve
        fitted_y = self.fit_curve()
        x_dense = np.linspace(transformed_x[0], transformed_x[-1], 200)
        self.ax.plot(x_dense, fitted_y, 'g--', alpha=0.5)

        # Plot all points
        if self.selected_point is not None:
            # First plot unselected points in red
            mask = np.ones(len(plot_points), dtype=bool)
            mask[self.selected_point] = False
            self.ax.plot(transformed_x[mask], plot_points[mask, 1], 'ro', markersize=8)
            
            # Then plot selected point in orange
            self.ax.plot(transformed_x[self.selected_point], 
                        plot_points[self.selected_point, 1], 
                        'o', color='#FF8C00', markersize=8)
        else:
            # If no point is selected, all points are red
            self.ax.plot(transformed_x, plot_points[:, 1], 'ro', markersize=8)

        # Set x-axis ticks
        self.ax.set_xticks([0, 1, 2, 3, 4])
        self.ax.set_xticklabels(['0', '10', '100', '1000', '15000'])

        # Set x-axis limits
        self.ax.set_xlim(-0.1, 4.2)

        # Compute percentage corresponding values
        percentages = [0, 20, 40, 60, 80, 100]
        values = [int(p * 255 / 100) for p in percentages]

        # Set y-axis ticks
        self.ax.set_yticks(values)
        self.ax2.set_yticks(values)

        # Display numerical values on the left
        self.ax.tick_params(axis='y', pad=10)  # Reduce distance from the chart
        self.ax.set_yticklabels(values)

        # Display percentage on the right
        self.ax2.tick_params(axis='y', pad=10)
        self.ax2.set_yticklabels([f"{p}%" for p in percentages])

        self.ax.grid(True)
        self.ax.set_xlabel('Ambient Light (lux)')
        self.ax.set_ylabel('Screen Brightness')

        self.canvas.draw()

    def find_nearest_point(self, event):
        """Find the nearest point to the mouse event."""
        if event.xdata is None or event.ydata is None:
            return None
        
        # Calculate distances to all points
        distances = np.sqrt(
            (self.transform_x_to_log(self.points[:, 0]) - event.xdata) ** 2 + 
            ((self.points[:, 1] - event.ydata) / 255) ** 2
        )
        
        nearest = np.argmin(distances)
        # Narrow the judgment range for precision
        if distances[nearest] < 0.035:  # Smaller threshold
            return nearest
        return None

    def is_near_curve(self, event):
        """Determine if the event is near the fitted curve."""
        if event.xdata is None or event.ydata is None:
            return False
        
        real_x = self.transform_x_from_log(event.xdata)
        
        # Interpolate y value at the clicked x position
        interpolated_y = np.interp(
            real_x,
            self.points[:, 0],
            self.points[:, 1]
        )
        
        distance = abs(interpolated_y - event.ydata)
        
        return distance < 10
    
    def add_point(self, x, y):
        """Add a new point at the specified location."""
        real_x = self.transform_x_from_log(x)
        
        # Narrow the minimum distance judgment
        log_distances = np.abs(
            np.log10(self.points[:, 0] + 0.1) - np.log10(real_x + 0.1)
        )
        if np.min(log_distances) < 0.01:  # Allow points to be closer
            return None
            
        new_point = np.array([[real_x, y]])
        self.points = np.vstack([self.points, new_point])
        self.points = self.points[self.points[:, 0].argsort()]  # Sort points by x value
        
        self.add_to_history()  # Save the state to history
        
        return np.where((self.points[:, 0] == real_x))[0][0]

    def fit_curve(self):
        """Smoothly fit the curve using UnivariateSpline."""
        x = self.points[:, 0]
        y = self.points[:, 1]
        
        # Convert to logarithmic space
        log_x = np.zeros_like(x)
        mask = x > 0
        log_x[mask] = np.log10(x[mask])
        
        # Increase s parameter value to avoid warnings
        spl = UnivariateSpline(log_x, y, s=len(x), k=3)
        
        # Generate denser points for a smoother curve
        x_dense = np.linspace(log_x[0], log_x[-1], 200)
        fitted_y = spl(x_dense)
        
        # Ensure results are within a reasonable range
        fitted_y = np.clip(fitted_y, 0, 255)
        
        # Ensure monotonicity
        for i in range(1, len(fitted_y)):
            fitted_y[i] = max(fitted_y[i], fitted_y[i-1])
        
        return fitted_y
    
    def auto_fit(self, event=None):
        """Automatically fit and redistribute points."""
        if len(self.points) < 2:
            return
            
        x = self.points[:, 0]
        y = self.points[:, 1]
        n_points = len(self.points)
        
        # Convert to logarithmic space
        log_x = np.zeros_like(x)
        mask = x > 0
        log_x[mask] = np.log10(x[mask])
        
        # Use spline fitting to get a smooth curve
        spl = UnivariateSpline(log_x, y, s=len(x), k=3)
        
        # Generate uniformly distributed new x values in logarithmic space
        log_x_start = log_x[0]
        log_x_end = log_x[-1]
        new_log_x = np.zeros_like(log_x)
        new_log_x[0] = log_x_start  # Keep starting point
        new_log_x[-1] = log_x_end    # Keep end point
        
        # Evenly distribute points in between
        step = (log_x_end - log_x_start) / (n_points - 1)
        for i in range(1, n_points-1):
            new_log_x[i] = log_x_start + step * i
        
        # Calculate new y values using fitted curve
        new_y = spl(new_log_x)
        
        # Convert back to actual x values
        new_x = 10 ** new_log_x
        new_x[0] = x[0]  # Ensure accuracy of starting point
        
        # Ensure results are within a reasonable range
        new_y = np.clip(new_y, 0, 255)
        new_y[0] = 0     # Ensure starting point value
        new_y[-1] = 255  # Ensure ending point value
        
        # Ensure monotonicity
        for i in range(1, len(new_y)):
            new_y[i] = max(new_y[i], new_y[i-1])
        
        # Update the points to their new locations
        self.points = np.column_stack((new_x, new_y))
        
        # Update history
        self.add_to_history()
        self.update_plot()

    def is_empty_area(self, event):
        """Determine if the click was in an empty area."""
        if event.xdata is None or event.ydata is None:
            return True
        
        # First check if the click is on a point
        if self.find_nearest_point(event) is not None:
            return False
        
        # Then check if it is near any line segment
        is_near, _ = self.is_near_line_segment(event)
        if is_near:
            return False
        
        return True

    def get_display_coord(self, x, y):
        """Convert data coordinates to display coordinates (pixels)."""
        return self.ax.transData.transform((x, y))

    def is_near_line_segment(self, event):
        """Determine if the click is near any line segment."""
        if event.xdata is None or event.ydata is None:
            return False, None

        # Get display coordinates of the click position
        click_coords = self.get_display_coord(event.xdata, event.ydata)
        click_x, click_y = click_coords

        min_distance = float('inf')
        best_segment = None
        best_t = None

        # Check each line segment
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            
            # Convert to logarithmic space
            log_x1 = np.log10(x1 + 0.1)
            log_x2 = np.log10(x2 + 0.1)
            
            # Get display coordinates of the segment endpoints
            p1_coords = self.get_display_coord(log_x1, y1)
            p2_coords = self.get_display_coord(log_x2, y2)
            x1_screen, y1_screen = p1_coords
            x2_screen, y2_screen = p2_coords

            # Calculate the shortest distance from the point to the line segment
            # Line segment vector
            line_vec = np.array([x2_screen - x1_screen, y2_screen - y1_screen])
            point_vec = np.array([click_x - x1_screen, click_y - y1_screen])
            line_length_sq = np.sum(line_vec * line_vec)
            
            if line_length_sq == 0:
                continue  # If the endpoints are the same, skip

            # Calculate the projection parameter t
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_sq))
            
            # Calculate the nearest point on the line segment
            proj_point = np.array([x1_screen, y1_screen]) + t * line_vec
            
            # Calculate actual distance
            distance = np.sqrt(np.sum((proj_point - [click_x, click_y])**2))
            
            # Update minimum distance
            if distance < min_distance:
                min_distance = distance
                best_segment = (log_x1, log_x2, y1, y2)
                best_t = t

        # Use a smaller threshold to determine if it is near
        if min_distance < 20:  # 20 pixels of acceptance range
            return True, best_segment

        return False, None

    def add_point_on_line(self, event, segment_info):
        """Precisely add a point on the line segment."""
        log_x1, log_x2, y1, y2 = segment_info
        
        # Calculate the precise position of the new point
        t = (event.xdata - log_x1) / (log_x2 - log_x1)
        new_y = y1 + t * (y2 - y1)
        
        # Add new point
        new_point_idx = self.add_point(event.xdata, new_y)
        return new_point_idx

    def on_click(self, event):
        """Handle click events."""
        if event.button != 1:
            return  # Only respond to left button clicks
                
        if event.xdata is None or event.ydata is None:
            self.selected_point = None
            self.info_label.config(text="")
            return
        
        # First try to select a point
        nearest = self.find_nearest_point(event)
        if nearest is not None:
            self.selected_point = nearest
            self.dragging = True
            point = self.points[self.selected_point]
            percentage = (point[1] * 100 / 255)
            self.info_label.config(
                text=f"Current: {point[0]:.1f} lux, {int(point[1])} brightness ({percentage:.1f}%)"
            )
            self.update_plot()
            return
        
        # Check if we are near a line segment and add a new point if so
        is_near, segment_info = self.is_near_line_segment(event)
        if is_near:
            new_point_idx = self.add_point_on_line(event, segment_info)
            if new_point_idx is not None:
                self.selected_point = new_point_idx
                self.dragging = True
                point = self.points[self.selected_point]
                percentage = (point[1] * 100 / 255)
                self.info_label.config(
                    text=f"Current: {point[0]:.1f} lux, {int(point[1])} brightness ({percentage:.1f}%)"
                )
                self.update_plot()
                return
        
        # If clicked in empty area, deselect
        self.selected_point = None
        self.info_label.config(text="")
        self.update_plot()
    
    def on_release(self, event):
        """Handle mouse release events."""
        if self.dragging:
            self.add_to_history()  # Save the state after dragging
        self.dragging = False
        self.update_plot()
    
    def on_motion(self, event):
        """Handle mouse motion events."""
        if self.dragging and self.selected_point is not None and event.xdata is not None:
            new_x = self.transform_x_from_log(event.xdata)
            new_y = max(0, min(255, event.ydata))
            
            # Ensure the new x position is within bounds of neighboring points
            if self.selected_point > 0:
                new_x = max(new_x, self.points[self.selected_point-1, 0])
            if self.selected_point < len(self.points)-1:
                new_x = min(new_x, self.points[self.selected_point+1, 0])
            
            self.points[self.selected_point] = [new_x, new_y]  # Update point position
            
            percentage = (new_y * 100 / 255)
            self.info_label.config(
                text=f"Current: {new_x:.1f} lux, {int(new_y)} brightness ({percentage:.1f}%)"
            )
            
            self.update_plot()
    
    def on_delete(self, event):
        """Handle delete key to remove selected point."""
        if self.selected_point is not None and len(self.points) > 2:
            self.points = np.delete(self.points, self.selected_point, axis=0)  # Remove selected point
            self.selected_point = None
            self.add_to_history()  # Log history after deletion
            self.update_plot()
    
    def on_closing(self):
        """Handle the window closing event."""
        plt.close('all')
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()
        self.root.destroy()
        sys.exit(0)
    
    def on_resize(self, event):
        """Handle window resize events."""
        # Redraw the chart
        self.update_plot()
    
    def export_parameters(self):
        """Export parameters to an editable text area."""
        export_window = tk.Toplevel(self.root)
        export_window.title("Parameters Editor")
        
        # Main frame
        main_frame = ttk.Frame(export_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text area frame
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text area
        text_area = tk.Text(text_frame, width=30, height=25)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate and insert parameter text
        params = []
        for x, y in self.points:
            params.append(f"Pt: {x:.4f} {int(y)}")
        param_string = '\n'.join(params)
        text_area.insert('1.0', param_string)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Status label frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Status label
        status_label = ttk.Label(status_frame, text="")
        status_label.pack(pady=(2, 0), anchor='w')  # 'w' means west (left)
        
        def update_parameters():
            """Update all parameters."""
            try:
                # Get text content
                text_content = text_area.get('1.0', tk.END).strip()
                lines = text_content.split('\n')
                
                # Parse parameters
                new_points = []
                for line in lines:
                    if not line.startswith('Pt:'):
                        continue
                        
                    try:
                        _, values = line.split(':', 1)
                        x_str, y_str = values.strip().split()
                        x = float(x_str)
                        y = int(float(y_str))
                        
                        # Ensure values are within valid range
                        if x < 0 or y < 0 or y > 255:
                            raise ValueError("Values out of range")
                            
                        new_points.append([x, y])  # Add the new point to the list
                    except:
                        continue  # Skip lines that cannot be parsed
                
                if len(new_points) < 2:
                    raise ValueError("Need at least 2 points")
                    
                # Update points
                self.points = np.array(new_points)
                self.points = self.points[self.points[:, 0].argsort()]  # Sort by x value
                
                # Update history
                self.add_to_history()
                
                # Update plot
                self.update_plot()
                
                # Show success message
                status_label.config(text="Parameters updated successfully!")
                export_window.after(1000, lambda: status_label.config(text=""))
                
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}")
                export_window.after(2000, lambda: status_label.config(text=""))
        
        def copy_parameters():
            """Copy parameters to clipboard."""
            text_content = text_area.get('1.0', tk.END).strip()
            export_window.clipboard_clear()
            export_window.clipboard_append(text_content)
            status_label.config(text="Parameters copied to clipboard!")
            export_window.after(1000, lambda: status_label.config(text=""))
        
        # Add buttons
        ttk.Button(button_frame, text="Update All", 
                command=update_parameters).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Copy Parameters", 
                command=copy_parameters).pack(side=tk.LEFT, padx=5)
        
        # Ensure child windows close when the main window closes
        export_window.transient(self.root)
        export_window.grab_set()

if __name__ == "__main__":
    root = tk.Tk()
    app = BrightnessCurveEditor(root)
    root.mainloop()