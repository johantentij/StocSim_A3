import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 1. Setup Data provided by User ---
snakes = np.array([[27, 7],
                   [50, 30],
                   [55, 35],
                   [68, 48],
                   [71, 61],
                   [77, 67],
                   [94, 84]])

ladders = np.array([[4, 25],
                    [10, 32],
                    [36, 52],
                    [43, 80],
                    [46, 66],
                    [63, 73],
                    [64, 83],
                    [75, 85]])

# --- 2. Helper Functions ---

def get_coords(number):
    """
    Converts a board number (1-100) into (x, y) coordinates
    for a 10x10 grid starting at bottom-left.
    Handles the Boustrophedon (snake-like) wrapping.
    """
    # Adjust to 0-indexed
    n = number - 1
    
    # Calculate row (y)
    row = n // 10
    
    # Calculate col (x)
    # If row is even (0, 2, 4...), go Left -> Right
    # If row is odd (1, 3, 5...), go Right -> Left
    if row % 2 == 0:
        col = n % 10
    else:
        col = 9 - (n % 10)
        
    return col, row

def draw_snake(ax, start, end):
    """Draws a curvy line to represent a snake"""
    p1 = get_coords(start)
    p2 = get_coords(end)
    
    # Create a curved path (Bezier-like logic using sin wave for style)
    x = np.linspace(p1[0], p2[0], 100)
    y = np.linspace(p1[1], p2[1], 100)
    
    # Add a curve offset
    curve_amp = 0.2
    curve = np.sin(np.linspace(0, 4*np.pi, 100)) * curve_amp
    
    # Adjust geometric orientation based on slope
    if p1[0] == p2[0]: # Vertical drop
        x += curve
    else:
        # Perpendicular offset for diagonal
        angle = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        x += curve * -np.sin(angle)
        y += curve * np.cos(angle)

    # Plot the snake body
    ax.plot(x + 0.5, y + 0.5, color='green', linewidth=4, solid_capstyle='round', alpha=0.8)
    # Interior stripe
    ax.plot(x + 0.5, y + 0.5, color='yellow', linewidth=1, alpha=0.8)
    
    # Draw Head and Tail markers
    ax.plot(p1[0]+0.5, p1[1]+0.5, 'o', color='darkgreen', markersize=8) # Head (Start)
    ax.plot(p2[0]+0.5, p2[1]+0.5, 'x', color='darkgreen', markersize=6) # Tail (End)

def draw_ladder(ax, start, end):
    """Draws a ladder structure"""
    p1 = get_coords(start)
    p2 = get_coords(end)
    
    # Coordinates centered in the square
    x1, y1 = p1[0] + 0.5, p1[1] + 0.5
    x2, y2 = p2[0] + 0.5, p2[1] + 0.5
    
    # Draw main rails (black lines)
    offset = 0.15
    ax.plot([x1-offset, x2-offset], [y1, y2], color='black', linewidth=2)
    ax.plot([x1+offset, x2+offset], [y1, y2], color='black', linewidth=2)
    
    # Draw rungs
    steps = 10
    for i in range(steps):
        ratio = i / (steps - 1)
        sx = (x1 - offset) * (1 - ratio) + (x2 - offset) * ratio
        ex = (x1 + offset) * (1 - ratio) + (x2 + offset) * ratio
        sy = y1 * (1 - ratio) + y2 * ratio
        ey = y1 * (1 - ratio) + y2 * ratio
        ax.plot([sx, ex], [sy, ey], color='black', linewidth=1.5)

# --- 3. Main Plotting Routine ---

def create_board():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors similar to the provided image (Yellow, Red, Blue, Greenish, White)
    colors = ['#FFD700', '#FF4500', '#1E90FF', '#32CD32', '#FFFFFF']
    
    # Draw the Grid Squares
    for i in range(1, 101):
        c, r = get_coords(i)
        
        # Pick color based on position (cycling pattern)
        color_idx = (c + r) % len(colors)
        rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='black', facecolor=colors[color_idx])
        ax.add_patch(rect)
        
        # Add Number Text
        # Highlight Start (1) and Finish (100)
        font_weight = 'normal'
        text_color = 'black'
        if i == 1:
            text_str = "1\nStart"
            font_weight = 'bold'
        elif i == 100:
            text_str = "100\nFinish"
            font_weight = 'bold'
        else:
            text_str = str(i)
            
        ax.text(c + 0.1, r + 0.8, text_str, fontsize=10, 
                ha='left', va='top', weight=font_weight, color=text_color)

    # Draw Ladders
    for start, end in ladders:
        draw_ladder(ax, start, end)

    # Draw Snakes
    for start, end in snakes:
        draw_snake(ax, start, end)

    plt.title("Snakes and Ladders Map", fontsize=16)
    plt.tight_layout()
    plt.show()

# Run the generation
if __name__ == "__main__":
    create_board()